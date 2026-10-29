-- battle_bridge.lua -- shared, game-agnostic bridge between mGBA and env.py.
-- Logic only; addresses come from a per-game config file.
--
-- Design rules (each one replaces a class of bugs from the old version):
--   1. "Are we in a battle" comes from the game's own flag (gMain.inBattle),
--      never from HP values. HP memory keeps stale values after a battle,
--      which caused input to leak into the overworld.
--   2. Outside a battle the bridge presses NOTHING (except optional
--      auto-walk, toggled with L). Every post-battle screen, evolution
--      included, runs while inBattle is still set.
--   3. Menus are navigated by reading the cursor position after every press,
--      not by timed key sequences. A press that doesn't land is retried, and
--      a stuck cursor fails the action cleanly instead of drifting.
--   4. env.py learns that a turn is over from the game's controller state
--      (STATUS command), not from "did HP change". No more blind re-sends.
--   5. Switch targets are identified by personality value (PID), because
--      the game reorders the party while the battle party menu is open.
--
-- Commands on port 8888 (one per line):
--   GET_STATE  -> comma-separated observation values (see getState)
--   STATUS     -> STATUS:<phase>:<lastResult>
--                 phase: NOT_IN_BATTLE, BUSY, CHOOSE_ACTION, FORCED_SWITCH,
--                        BATTLE_OVER, WAITING
--                 lastResult: NONE, PENDING, OK, FAILED, ABORTED
--   ACTION:n   -> ACK, or REJECT:<reason>
-- Port 8889 serves the memory-address finder (SIGNALS, DUMP).

local GAME_CONFIG_FILE = "/home/kazuto/Documents/Projects/PokeGen3-RL/games/emerald/emerald_config.lua"
local CONFIG = dofile(GAME_CONFIG_FILE)

local MAIN_PORT = 8888
local PROBE_PORT = 8889
local LOG_CALLBACK2_CHANGES = true

local KEY = C.GBA_KEY
local ALL_KEYS = 0x3FF

-- ===================================================================== state

local server, client, clientBuf = nil, nil, ""
local probeServer, probeClient, probeBuf = nil, nil, ""
local frameCount = 0

-- Current bridge-driven action, or nil.
local task = nil
local lastResult = "NONE"
local TASK_MAX_FRAMES = 1200 -- 20 s: generous, a normal action takes < 3 s

-- Party order reported to env.py: active mon first, then the rest in field
-- order. reportedPids[k] is the PID in reported slot k-1 (0-based slots).
local fieldOrderPids = {}
local reportedPids = {}

-- One key at a time: press for PRESS_FRAMES, release for RELEASE_FRAMES so
-- every tap reaches the game as a fresh press (menus check JOY_NEW).
local PRESS_FRAMES, RELEASE_FRAMES = 2, 4
local input = { key = nil, stage = 0, timer = 0 }

-- In-battle prompt automation
local promptCooldown = 0
local watchdog = { sig = nil, frames = 0 }
local WATCHDOG_FRAMES = 600 -- 10 s stuck on the same screen -> nudge with B

-- Auto-walk (overworld only, toggled with L)
local autoWalkEnabled, walkStep, walkTimer, lastToggleHeld, toggleCooldown = false, 0, 0, false, 0
local WALK_SEQUENCE = { KEY.UP, KEY.RIGHT, KEY.DOWN, KEY.LEFT }

local lastLoggedCb2 = nil
local wasInBattle = false

-- Gen 3 party substructure orders, keyed by PID % 24 (same for all Gen 3).
local block_orders = {
	{ 0, 1, 2, 3 },
	{ 0, 1, 3, 2 },
	{ 0, 2, 1, 3 },
	{ 0, 2, 3, 1 },
	{ 0, 3, 1, 2 },
	{ 0, 3, 2, 1 },
	{ 1, 0, 2, 3 },
	{ 1, 0, 3, 2 },
	{ 1, 2, 0, 3 },
	{ 1, 2, 3, 0 },
	{ 1, 3, 0, 2 },
	{ 1, 3, 2, 0 },
	{ 2, 0, 1, 3 },
	{ 2, 0, 3, 1 },
	{ 2, 1, 0, 3 },
	{ 2, 1, 3, 0 },
	{ 2, 3, 0, 1 },
	{ 2, 3, 1, 0 },
	{ 3, 0, 1, 2 },
	{ 3, 0, 2, 1 },
	{ 3, 1, 0, 2 },
	{ 3, 1, 2, 0 },
	{ 3, 2, 0, 1 },
	{ 3, 2, 1, 0 },
}

-- ============================================================ memory helpers

-- Function pointers in RAM carry the Thumb bit (bit 0); the .sym file lists
-- the even address. Compare with bit 0 ignored.
local function sameFn(a, b)
	if a == nil or b == nil then
		return false
	end
	return (a | 1) == (b | 1)
end

local function inRom(p)
	return p >= 0x08000000 and p < 0x0A000000
end

local function readCb2()
	return emu:read32(CONFIG.MAIN_CALLBACK2_ADDR)
end

local function isInBattle()
	return (emu:read8(CONFIG.GMAIN_INBATTLE_ADDR) & CONFIG.GMAIN_INBATTLE_MASK) ~= 0
end

local function battleOutcome()
	return emu:read8(CONFIG.BATTLE_OUTCOME_ADDR)
end

local function controllerFn(battler)
	return emu:read32(CONFIG.CONTROLLER_FUNCS_ADDR + battler * 4)
end

local function scriptOpcode()
	local p = emu:read32(CONFIG.BATTLESCRIPT_PTR_ADDR)
	if inRom(p) then
		return emu:read8(p)
	end
	return nil
end

local function isDoubleBattle()
	return (emu:read32(CONFIG.BATTLE_TYPE_FLAGS_ADDR) & CONFIG.BATTLE_TYPE_DOUBLE_BIT) ~= 0
end

local function battlerBase(battler)
	return CONFIG.BATTLE_MONS_ADDR + battler * CONFIG.BATTLE_MON_STRIDE
end

local function isInputHandler(fn)
	return sameFn(fn, CONFIG.FN_CHOOSE_ACTION)
		or sameFn(fn, CONFIG.FN_CHOOSE_MOVE)
		or sameFn(fn, CONFIG.FN_CHOOSE_TARGET)
end

local function partySlotBase(slot)
	return CONFIG.PARTY_START + slot * CONFIG.PARTY_STRIDE
end

local function memoryPids()
	local t = {}
	for i = 0, 5 do
		t[i] = emu:read32(partySlotBase(i))
	end
	return t
end

local function memIndexOf(pid, mem)
	if pid == nil or pid == 0 then
		return nil
	end
	for i = 0, 5 do
		if mem[i] == pid then
			return i
		end
	end
	return nil
end

-- ===================================================================== input

local function inputBusy()
	return input.stage ~= 0
end

local function tap(key)
	if input.stage ~= 0 then
		return false
	end
	input.key, input.stage, input.timer = key, 1, 0
	emu:addKey(key)
	return true
end

local function updateInput()
	if input.stage == 1 then
		input.timer = input.timer + 1
		if input.timer >= PRESS_FRAMES then
			emu:clearKeys(ALL_KEYS)
			input.stage, input.timer = 2, 0
		else
			emu:addKey(input.key)
		end
	elseif input.stage == 2 then
		input.timer = input.timer + 1
		if input.timer >= RELEASE_FRAMES then
			input.stage, input.timer, input.key = 0, 0, nil
		end
	end
end

local function cancelInput()
	emu:clearKeys(ALL_KEYS)
	input.stage, input.timer, input.key = 0, 0, nil
end

-- Direction that moves a 2x2 grid cursor (bit 0 = column, bit 1 = row)
-- one step toward target, or nil when already there.
local function gridStep(cur, target)
	if (cur & 1) ~= (target & 1) then
		return ((target & 1) == 1) and KEY.RIGHT or KEY.LEFT
	end
	if (cur & 2) ~= (target & 2) then
		return ((target & 2) == 2) and KEY.DOWN or KEY.UP
	end
	return nil
end

-- ================================================================ party data

local function decryptPartySlot(base)
	local pid = emu:read32(base)
	if pid == 0 then
		return nil
	end
	local key = pid ~ emu:read32(base + 4)
	local order = block_orders[(pid % 24) + 1]
	local function blockAddr(t)
		for i = 1, 4 do
			if order[i] == t then
				return base + 0x20 + (i - 1) * 12
			end
		end
	end
	local g, a = blockAddr(0), blockAddr(1)
	local g0 = emu:read32(g) ~ key
	local a0 = emu:read32(a) ~ key
	local a1 = emu:read32(a + 4) ~ key
	return {
		species = g0 & 0xFFFF,
		item = g0 >> 16,
		moves = { a0 & 0xFFFF, a0 >> 16, a1 & 0xFFFF, a1 >> 16 },
		hp = emu:read16(base + CONFIG.PARTY_HP_OFFSET),
		maxhp = emu:read16(base + CONFIG.PARTY_MAXHP_OFFSET),
	}
end

-- Memory holds the party in field order except while the in-battle party
-- menu is open, so only refresh the cache while BattleMainCB2 runs.
local function refreshFieldOrder(cb2)
	if not sameFn(cb2, CONFIG.CB2_BATTLE_MAIN) then
		return
	end
	local t = {}
	for i = 0, 5 do
		local pid = emu:read32(partySlotBase(i))
		if pid ~= 0 then
			t[#t + 1] = pid
		end
	end
	fieldOrderPids = t
end

local function buildReportedOrder()
	local mem = memoryPids()
	local activePid = emu:read32(battlerBase(0) + CONFIG.BATTLER_OFFSETS.personality)
	local order, used = {}, {}
	local function add(pid)
		if pid ~= 0 and not used[pid] and memIndexOf(pid, mem) then
			order[#order + 1] = pid
			used[pid] = true
		end
	end
	add(activePid)
	for _, pid in ipairs(fieldOrderPids) do
		add(pid)
	end
	for i = 0, 5 do
		add(mem[i])
	end -- anything the cache missed
	return order, mem
end

-- ================================================================ GET_STATE

-- 21 values per battler, in the order env.py's _parse_battler() unpacks.
local function getBattler(base)
	local o = CONFIG.BATTLER_OFFSETS
	return {
		emu:read16(base + o.hp),
		emu:read16(base + o.max_hp),
		emu:read8(base + o.pp1),
		emu:read8(base + o.pp2),
		emu:read8(base + o.pp3),
		emu:read8(base + o.pp4),
		emu:read16(base + o.move1),
		emu:read16(base + o.move2),
		emu:read16(base + o.move3),
		emu:read16(base + o.move4),
		emu:read8(base + o.type1),
		emu:read8(base + o.type2),
		emu:read8(base + o.level),
		emu:read32(base + o.status1),
		emu:read32(base + o.status2),
		emu:read8(base + o.atk_buff),
		emu:read8(base + o.def_buff),
		emu:read8(base + o.spd_buff),
		emu:read16(base + o.species),
		emu:read16(base + o.item),
		emu:read8(base + o.ability),
	}
end

local EMPTY_BATTLER = { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 }

local function weatherCategory()
	local w = emu:read16(CONFIG.WEATHER_ADDR)
	if (w & CONFIG.WEATHER_RAIN_MASK) ~= 0 then
		return 1
	end
	if (w & CONFIG.WEATHER_SAND_MASK) ~= 0 then
		return 2
	end
	if (w & CONFIG.WEATHER_SUN_MASK) ~= 0 then
		return 3
	end
	if (w & CONFIG.WEATHER_HAIL_MASK) ~= 0 then
		return 4
	end
	return 0
end

local computePhase -- defined below

function getState()
	local b0 = getBattler(CONFIG.BATTLER_BASE_P1)
	local b1 = getBattler(CONFIG.BATTLER_BASE_E1)
	local b2 = getBattler(CONFIG.BATTLER_BASE_P2)
	local b3 = getBattler(CONFIG.BATTLER_BASE_E2)
	if not isDoubleBattle() then
		b2, b3 = EMPTY_BATTLER, EMPTY_BATTLER
	end

	local inBattle = isInBattle()
	local outcome = battleOutcome()
	local battleOver = (inBattle and outcome ~= 0) and 1 or 0
	local playerLost = 0
	if battleOver == 1 and (outcome == CONFIG.OUTCOME_LOST or outcome == CONFIG.OUTCOME_FORFEITED) then
		playerLost = 1
	end

	-- Player party: active mon first, then field order (see buildReportedOrder).
	local order, mem = buildReportedOrder()
	reportedPids = order
	local p_party = {}
	for k = 1, 6 do
		local slot = nil
		local idx = memIndexOf(order[k], mem)
		if idx then
			slot = decryptPartySlot(partySlotBase(idx))
		end
		if slot then
			p_party[#p_party + 1] = slot.hp
			p_party[#p_party + 1] = slot.maxhp
			p_party[#p_party + 1] = slot.species
			p_party[#p_party + 1] = slot.item
			for m = 1, 4 do
				p_party[#p_party + 1] = slot.moves[m]
			end
		else
			for _ = 1, 8 do
				p_party[#p_party + 1] = 0
			end
		end
	end

	local e_party = {}
	for i = 0, 5 do
		local e = decryptPartySlot(CONFIG.ENEMY_PARTY_START + i * CONFIG.PARTY_STRIDE)
		if e then
			e_party[#e_party + 1] = e.species
			e_party[#e_party + 1] = e.item
			e_party[#e_party + 1] = e.hp
			e_party[#e_party + 1] = e.maxhp
			for m = 1, 4 do
				e_party[#e_party + 1] = e.moves[m]
			end
		else
			for _ = 1, 8 do
				e_party[#e_party + 1] = 0
			end
		end
	end

	local phase = computePhase()
	local out = {}
	for _, list in ipairs({ b0, b1, b2, b3 }) do
		for _, v in ipairs(list) do
			out[#out + 1] = v
		end
	end
	out[#out + 1] = battleOver
	out[#out + 1] = playerLost
	for _, v in ipairs(p_party) do
		out[#out + 1] = v
	end
	for _, v in ipairs(e_party) do
		out[#out + 1] = v
	end
	out[#out + 1] = weatherCategory() -- 0 none, 1 rain, 2 sand, 3 sun, 4 hail
	out[#out + 1] = 0 -- activeSlot: always 0 now (active is reported first)
	out[#out + 1] = (phase == "FORCED_SWITCH") and 1 or 0
	out[#out + 1] = outcome -- raw gBattleOutcome
	return table.concat(out, ",")
end

-- ===================================================================== phase

-- Which player battler is waiting for input (0, or 2 in doubles), or nil.
local function choosingBattler()
	if isInputHandler(controllerFn(0)) then
		return 0
	end
	if isDoubleBattle() and isInputHandler(controllerFn(2)) then
		return 2
	end
	return nil
end

computePhase = function()
	if not isInBattle() then
		return "NOT_IN_BATTLE", nil
	end
	if task then
		return "BUSY", task.battler
	end
	if battleOutcome() ~= 0 then
		return "BATTLE_OVER", nil
	end
	local cb2 = readCb2()
	if
		sameFn(cb2, CONFIG.CB2_PARTY_MENU)
		and emu:read8(CONFIG.PARTY_MENU_ACTION_ADDR) == CONFIG.PARTY_ACTION_SEND_OUT
	then
		return "FORCED_SWITCH", 0
	end
	if sameFn(cb2, CONFIG.CB2_BATTLE_MAIN) then
		local b = choosingBattler()
		if b ~= nil and not sameFn(controllerFn(b), CONFIG.FN_CHOOSE_TARGET) then
			return "CHOOSE_ACTION", b
		end
	end
	return "WAITING", nil
end

-- ===================================================================== tasks

local function finishTask(result, why)
	if result ~= "OK" then
		console:log(string.format("[bridge] %s %s: %s", task and task.kind or "?", result, why or ""))
	end
	lastResult = result
	task = nil
end

local function newTask(kind, battler)
	return {
		kind = kind,
		battler = battler,
		stage = "start",
		frames = 0,
		cooldown = 0,
		navKey = nil,
		navCount = 0,
		aPresses = 0,
		waitFrames = 0,
	}
end

-- Tap a direction; fail the task if the cursor refuses to move after
-- several taps (e.g. trying to reach a move slot that doesn't exist).
local function navTap(key, cursorValue)
	local sig = tostring(key) .. ":" .. tostring(cursorValue)
	if task.navKey == sig then
		task.navCount = task.navCount + 1
	else
		task.navKey, task.navCount = sig, 1
	end
	if task.navCount > 5 then
		finishTask("FAILED", "cursor stuck at " .. tostring(cursorValue))
		return
	end
	tap(key)
	task.cooldown = 2
end

local function runMoveTask(cb2)
	if not sameFn(cb2, CONFIG.CB2_BATTLE_MAIN) then
		return
	end
	local b = task.battler
	local ctrl = controllerFn(b)

	if sameFn(ctrl, CONFIG.FN_CHOOSE_ACTION) then
		local cur = emu:read8(CONFIG.ACTION_CURSOR_ADDR + b)
		local dir = gridStep(cur, 0)
		if dir then
			navTap(dir, cur)
		else
			tap(KEY.A)
			task.stage = "fight_pressed"
			task.cooldown = 6
		end
	elseif sameFn(ctrl, CONFIG.FN_CHOOSE_MOVE) then
		local cur = emu:read8(CONFIG.MOVE_CURSOR_ADDR + b)
		local dir = gridStep(cur, task.moveIdx)
		if dir then
			navTap(dir, cur)
		else
			task.aPresses = task.aPresses + 1
			if task.aPresses > 3 then
				-- The game keeps refusing the move (no PP, disabled, ...).
				finishTask("FAILED", "move " .. task.moveIdx .. " refused by game")
				return
			end
			tap(KEY.A)
			task.stage = "move_pressed"
			task.cooldown = 30 -- room for a "no PP left" message
		end
	elseif sameFn(ctrl, CONFIG.FN_CHOOSE_TARGET) then
		local cur = emu:read8(CONFIG.TARGET_CURSOR_ADDR)
		if cur ~= task.targetBattler then
			navTap(KEY.RIGHT, cur)
		else
			tap(KEY.A)
			task.stage = "target_pressed"
			task.cooldown = 6
		end
	elseif task.stage ~= "start" then
		-- The controller left the input handlers: the game took the choice
		-- (this also covers Struggle, where FIGHT skips the move menu).
		finishTask("OK")
	end
end

local function runSwitchTask(cb2)
	local b = task.battler
	if sameFn(cb2, CONFIG.CB2_BATTLE_MAIN) then
		local ctrl = controllerFn(b)
		if task.stage == "confirm_pressed" then
			if sameFn(ctrl, CONFIG.FN_CHOOSE_ACTION) then
				finishTask("FAILED", "switch did not happen")
			elseif not isInputHandler(ctrl) then
				finishTask("OK")
			end
			return
		end
		if task.kind == "forced" then
			return
		end -- waiting for the party menu
		if sameFn(ctrl, CONFIG.FN_CHOOSE_MOVE) then
			tap(KEY.B) -- back to the action menu
			task.cooldown = 6
		elseif sameFn(ctrl, CONFIG.FN_CHOOSE_ACTION) then
			local cur = emu:read8(CONFIG.ACTION_CURSOR_ADDR + b)
			local dir = gridStep(cur, 2) -- 2 = POKEMON
			if dir then
				navTap(dir, cur)
			else
				tap(KEY.A)
				task.stage = "pokemon_pressed"
				task.cooldown = 10
			end
		end
		return
	end

	if not sameFn(cb2, CONFIG.CB2_PARTY_MENU) then
		return
	end -- fades/transitions

	local partyAction = emu:read8(CONFIG.PARTY_MENU_ACTION_ADDR)
	local function giveUp(why)
		if partyAction ~= CONFIG.PARTY_ACTION_SEND_OUT then
			tap(KEY.B) -- leave the menu; B does nothing on a forced send-out
		end
		finishTask("FAILED", why)
	end

	if task.stage == "confirm_pressed" then
		-- Still in the menu after confirming: the game refused the choice
		-- (fainted, already in battle, ...). Give it time for the message.
		task.waitFrames = task.waitFrames + 1
		if task.waitFrames > 180 then
			giveUp("party menu refused slot")
		end
		return
	end
	if task.stage == "slot_pressed" then
		tap(KEY.A) -- first item of the submenu: SHIFT / SEND OUT
		task.stage = "confirm_pressed"
		task.waitFrames = 0
		task.cooldown = 20
		return
	end

	local j = memIndexOf(task.pid, memoryPids())
	if j == nil then
		giveUp("target Pokemon not found in party")
		return
	end
	local cur = emu:read8(CONFIG.PARTY_MENU_SLOT_ADDR)
	if cur ~= j then
		navTap((cur > j) and KEY.UP or KEY.DOWN, cur)
	else
		tap(KEY.A)
		task.stage = "slot_pressed"
		task.cooldown = 20
	end
end

local function runTask()
	task.frames = task.frames + 1
	if not isInBattle() then
		finishTask("ABORTED", "battle ended")
		return
	end
	if task.frames > TASK_MAX_FRAMES then
		finishTask("FAILED", "timed out in stage " .. task.stage)
		return
	end
	if inputBusy() then
		return
	end
	if task.cooldown > 0 then
		task.cooldown = task.cooldown - 1
		return
	end
	local cb2 = readCb2()
	if task.kind == "move" then
		runMoveTask(cb2)
	else
		runSwitchTask(cb2)
	end
end

-- action 0-11: move (action % 4) on target (action // 4: 0 enemy left,
-- 1 enemy right, 2 partner). action 12-16: switch to reported slot 1-5.
local function handleAction(action)
	if action == nil or action < 0 or action > 16 then
		return "REJECT:bad_action"
	end
	if task then
		return "REJECT:busy"
	end
	local phase, battler = computePhase()

	if phase == "CHOOSE_ACTION" then
		if action < 12 then
			task = newTask("move", battler)
			task.moveIdx = action % 4
			local targetId = action // 4
			if targetId == 0 then
				task.targetBattler = 1
			elseif targetId == 1 then
				task.targetBattler = 3
			else
				task.targetBattler = (battler == 0) and 2 or 0
			end
		else
			local pid = reportedPids[action - 11 + 1]
			if not pid then
				return "REJECT:empty_slot"
			end
			task = newTask("switch", battler)
			task.pid = pid
		end
	elseif phase == "FORCED_SWITCH" then
		if action < 12 then
			return "REJECT:switch_required"
		end
		local pid = reportedPids[action - 11 + 1]
		if not pid then
			return "REJECT:empty_slot"
		end
		task = newTask("forced", 0)
		task.pid = pid
	else
		return "REJECT:" .. phase
	end
	lastResult = "PENDING"
	return "ACK"
end

-- ========================================================= prompt automation

-- Handles every in-battle screen that is not the model's decision: level-up
-- box, "learn a new move?" (always declined), "use next Pokemon?" (yes),
-- "change Pokemon?" when the enemy switches (no), evolution (let it happen).
local function battleAutomation(cb2)
	if inputBusy() then
		return
	end
	if promptCooldown > 0 then
		promptCooldown = promptCooldown - 1
		return
	end

	if sameFn(cb2, CONFIG.CB2_EVOLUTION_LOAD) or sameFn(cb2, CONFIG.CB2_EVOLUTION_UPDATE) then
		tap(KEY.A) -- never B: B cancels the evolution
		promptCooldown = 15
		watchdog.sig, watchdog.frames = nil, 0
		return
	end

	if sameFn(cb2, CONFIG.CB2_BATTLE_MAIN) then
		local op = scriptOpcode()
		local ctrl0 = controllerFn(0)
		if op == CONFIG.OP_LEARNMOVE_YESNO then
			tap(KEY.B) -- "Delete a move?" -> No
			promptCooldown = 20
		elseif op == CONFIG.OP_STOPLEARNING_YESNO then
			tap(KEY.A) -- "Stop learning?" -> Yes (B here would loop back)
			promptCooldown = 20
		elseif op == CONFIG.OP_YESNOBOX then
			if emu:read16(battlerBase(0) + CONFIG.BATTLER_OFFSETS.hp) == 0 then
				tap(KEY.A) -- our mon fainted: "Use next Pokemon?" -> Yes
			else
				tap(KEY.B) -- enemy is switching: "Change Pokemon?" -> No
			end
			promptCooldown = 20
		elseif op == CONFIG.OP_LVLUP_BOX then
			tap(KEY.B)
			promptCooldown = 20
		elseif sameFn(ctrl0, CONFIG.FN_YESNO_INPUT) then
			tap(KEY.B)
			promptCooldown = 20
		elseif choosingBattler() == nil then
			-- Nothing recognised and the game isn't waiting for the model.
			-- If it sits on the same script opcode for a long time, it is
			-- probably waiting for a button we don't know about: nudge with B
			-- and log it so the prompt can be identified and handled properly.
			local sig = tostring(op) .. ":" .. tostring(ctrl0)
			if sig == watchdog.sig then
				watchdog.frames = watchdog.frames + 1
			else
				watchdog.sig, watchdog.frames = sig, 0
			end
			if watchdog.frames > WATCHDOG_FRAMES then
				console:log(
					string.format(
						"[bridge] watchdog: pressed B (opcode %s, ctrl 0x%08X)",
						op and string.format("0x%02X", op) or "nil",
						ctrl0
					)
				)
				tap(KEY.B)
				watchdog.frames = WATCHDOG_FRAMES - 120
			end
			return
		end
		watchdog.sig, watchdog.frames = nil, 0
		return
	end

	-- Any other screen while still in battle, after the battle is decided:
	-- e.g. the summary screen that evolution's "forget a move?" opens. Back
	-- out with B (keeps the existing moves).
	if battleOutcome() ~= 0 and not sameFn(cb2, CONFIG.CB2_PARTY_MENU) then
		tap(KEY.B)
		promptCooldown = 30
	end
end

-- ================================================================= auto-walk

local function checkToggleKey()
	if toggleCooldown > 0 then
		toggleCooldown = toggleCooldown - 1
		return
	end
	local held = (emu:getKey(KEY.L) == 1)
	if held and not lastToggleHeld then
		autoWalkEnabled = not autoWalkEnabled
		console:log("Auto-walk: " .. (autoWalkEnabled and "ON" or "OFF"))
		if not autoWalkEnabled then
			cancelInput()
		end
		toggleCooldown = 20
	end
	lastToggleHeld = held
end

local function processWalk(cb2)
	if not autoWalkEnabled or not sameFn(cb2, CONFIG.CB2_OVERWORLD) then
		return
	end
	walkTimer = walkTimer + 1
	emu:addKey(WALK_SEQUENCE[walkStep + 1])
	if walkTimer >= 15 then
		emu:clearKeys(ALL_KEYS)
		walkStep = (walkStep + 1) % 4
		walkTimer = 0
	end
end

-- ============================================================ finder (probe)

local function signalsLine()
	local phase = computePhase()
	local opcode = isInBattle() and scriptOpcode() or nil
	return string.format(
		"frame=%d;cb2=0x%08X;inBattle=%d;outcome=%d;ctrl0=0x%08X;ctrl2=0x%08X;"
			.. "actionCursor=%d;moveCursor=%d;targetCursor=%d;partySlot=%d;partyAction=%d;"
			.. "opcode=%d;battleType=0x%08X;pHP=%d;eHP=%d;weather=%d;phase=%s;task=%s;key=%d",
		frameCount,
		readCb2(),
		isInBattle() and 1 or 0,
		battleOutcome(),
		controllerFn(0),
		controllerFn(2),
		emu:read8(CONFIG.ACTION_CURSOR_ADDR),
		emu:read8(CONFIG.MOVE_CURSOR_ADDR),
		emu:read8(CONFIG.TARGET_CURSOR_ADDR),
		emu:read8(CONFIG.PARTY_MENU_SLOT_ADDR),
		emu:read8(CONFIG.PARTY_MENU_ACTION_ADDR),
		opcode or -1,
		emu:read32(CONFIG.BATTLE_TYPE_FLAGS_ADDR),
		emu:read16(battlerBase(0) + CONFIG.BATTLER_OFFSETS.hp),
		emu:read16(battlerBase(1) + CONFIG.BATTLER_OFFSETS.hp),
		emu:read16(CONFIG.WEATHER_ADDR),
		phase,
		task and task.kind or "none",
		input.key or -1
	)
end

local function readRegion(start, size)
	local ok, data = pcall(function()
		return emu:readRange(start, size)
	end)
	if ok and type(data) == "string" and #data == size then
		return data
	end
	-- Fallback for builds without readRange: 4 bytes at a time.
	local parts = {}
	for a = start, start + size - 1, 4 do
		parts[#parts + 1] = string.pack("<I4", emu:read32(a))
	end
	return table.concat(parts)
end

-- Outgoing probe data is queued and sent in chunks across frames, so a
-- 288 KB dump never stalls the emulator or gets cut off by a full buffer.
local probeOut, probeOutPos, probeStall = "", 1, 0
local PROBE_CHUNK = 32768

local function probeQueue(data)
	if probeOutPos > #probeOut then
		probeOut, probeOutPos = data, 1
	else
		probeOut = string.sub(probeOut, probeOutPos) .. data
		probeOutPos = 1
	end
end

local function probeFlush()
	if not probeClient or probeOutPos > #probeOut then
		return
	end
	for _ = 1, 8 do
		if probeOutPos > #probeOut then
			break
		end
		local chunk = string.sub(probeOut, probeOutPos, probeOutPos + PROBE_CHUNK - 1)
		local n = probeClient:send(chunk)
		if n == true then
			n = #chunk
		end -- some builds return true instead of a count
		if type(n) ~= "number" or n <= 0 then
			probeStall = probeStall + 1
			if probeStall > 600 then
				console:log("address finder not reading, dropping it")
				probeClient, probeOut, probeOutPos, probeStall = nil, "", 1, 0
			end
			return
		end
		probeStall = 0
		probeOutPos = probeOutPos + n
	end
end

local function handleProbeLine(line)
	if line == "SIGNALS" then
		probeQueue(signalsLine() .. "\n")
	elseif line == "DUMP" then
		local parts, total = {}, 0
		for _, r in ipairs(CONFIG.RAM_REGIONS) do
			local data = readRegion(r.start, r.size)
			parts[#parts + 1] = data
			total = total + #data
		end
		probeQueue(string.format("DUMP %d %s\n", total, signalsLine()) .. table.concat(parts))
	elseif line == "REGIONS" then
		local t = {}
		for _, r in ipairs(CONFIG.RAM_REGIONS) do
			t[#t + 1] = string.format("%s:0x%08X:%d", r.name, r.start, r.size)
		end
		probeQueue(table.concat(t, ",") .. "\n")
	end
end

-- ================================================================== sockets

local function startServers()
	server = socket.bind(nil, MAIN_PORT)
	server:listen()
	probeServer = socket.bind(nil, PROBE_PORT)
	if probeServer then
		probeServer:listen()
	end
	console:log(string.format("Bridge listening on %d (env) and %d (address finder)", MAIN_PORT, PROBE_PORT))
end

-- Reads available data into buf and returns complete lines (handles
-- partial and multiple lines per receive, which receive(1024) alone didn't).
local function readLines(sock, buf)
	local lines = {}
	if not sock:hasdata() then
		return lines, buf, true
	end
	local chunk = sock:receive(1024)
	if chunk == nil or chunk == "" then
		return lines, buf, false
	end
	buf = buf .. chunk
	while true do
		local nl = string.find(buf, "\n", 1, true)
		if not nl then
			break
		end
		local line = string.sub(buf, 1, nl - 1):gsub("%s+$", "")
		buf = string.sub(buf, nl + 1)
		if line ~= "" then
			lines[#lines + 1] = line
		end
	end
	return lines, buf, true
end

local function handleMainLine(line)
	if line == "GET_STATE" then
		client:send(getState() .. "\n")
	elseif line == "STATUS" then
		local phase = computePhase()
		client:send("STATUS:" .. phase .. ":" .. lastResult .. "\n")
	elseif string.match(line, "^ACTION:") then
		local action = tonumber(string.match(line, "^ACTION:(%d+)"))
		client:send(handleAction(action) .. "\n")
	end
end

local function serviceSockets()
	if not client then
		local c = server:accept()
		if c then
			client, clientBuf = c, ""
			console:log("env.py connected")
		end
	else
		local lines, buf, alive = readLines(client, clientBuf)
		clientBuf = buf
		if not alive then
			console:log("env.py disconnected")
			client = nil
		else
			for _, line in ipairs(lines) do
				handleMainLine(line)
			end
		end
	end

	if probeServer then
		if not probeClient then
			local c = probeServer:accept()
			if c then
				probeClient, probeBuf = c, ""
				probeOut, probeOutPos, probeStall = "", 1, 0
				console:log("address finder connected")
			end
		else
			local lines, buf, alive = readLines(probeClient, probeBuf)
			probeBuf = buf
			if not alive then
				console:log("address finder disconnected")
				probeClient = nil
			else
				for _, line in ipairs(lines) do
					handleProbeLine(line)
				end
				probeFlush()
			end
		end
	end
end

-- ==================================================================== frame

function onFrame()
	frameCount = frameCount + 1
	updateInput()
	checkToggleKey()

	local cb2 = readCb2()
	local inBattle = isInBattle()

	if LOG_CALLBACK2_CHANGES and cb2 ~= lastLoggedCb2 then
		lastLoggedCb2 = cb2
		console:log(string.format("callback2 -> 0x%08X%s", cb2, inBattle and " (in battle)" or ""))
	end

	if inBattle and not wasInBattle then
		-- Release anything auto-walk was holding so it can't move a menu cursor.
		cancelInput()
		walkTimer = 0
	end
	wasInBattle = inBattle

	if inBattle then
		refreshFieldOrder(cb2)
		if task then
			runTask()
		else
			battleAutomation(cb2)
		end
	else
		if task then
			finishTask("ABORTED", "not in battle")
		end
		if inputBusy() and not autoWalkEnabled then
			cancelInput()
		end
		watchdog.sig, watchdog.frames, promptCooldown = nil, 0, 0
		processWalk(cb2)
	end

	serviceSockets()
end

callbacks:add("frame", onFrame)
startServers()
