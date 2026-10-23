local server = nil
local client = nil

-- Global Addresses needed for the Action/Walk loops
local ADDR_PLAYER_HP = 0x020240AC
local ADDR_ENEMY_HP = 0x02024104

local PARTY_MENU_SLOT_ADDR = 0x0203CED1
local PARTY_HP_BASE = 0x02024542
local PARTY_START = 0x020244EC
local PARTY_STRIDE = 0x64

local pendingAction, actionStep, actionTimer = nil, 0, 0
local pendingSwitch, switchTargetSlot, switchStep, switchTimer = nil, 0, 0, 0
local autoWalkEnabled, walkStep, walkTimer, lastSelectState, toggleCooldown = false, 0, 0, false, 0

local WALK_SEQUENCE = { C.GBA_KEY.UP, C.GBA_KEY.RIGHT, C.GBA_KEY.DOWN, C.GBA_KEY.LEFT }
local EMPTY_SLOT = { species = 0, item = 0, moves = { 0, 0, 0, 0 } }
local EMPTY_ENEMY_SLOT = { species = 0, item = 0, hp = 0, maxhp = 0, moves = { 0, 0, 0, 0 } }

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

function startServer()
	server = socket.bind(nil, 8888)
	server:listen()
	console:log("Lua server listening on port 8888")
end

function readPartySlot(slot)
	local base = PARTY_START + slot * PARTY_STRIDE
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
	}
end

function readEnemyPartySlot(slot)
	local base = 0x02024744 + slot * PARTY_STRIDE
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
		hp = emu:read16(base + 86),
		maxhp = emu:read16(base + 88),
	}
end

function getBattler(base)
	-- Returns 21 exact variables per active battler based on struct offsets
	return {
		emu:read16(base + 0x28),
		emu:read16(base + 0x2A), -- 1,2: hp, maxhp
		emu:read8(base + 0x24),
		emu:read8(base + 0x25),
		emu:read8(base + 0x26),
		emu:read8(base + 0x27), -- 3-6: pp
		emu:read16(base + 0x0C),
		emu:read16(base + 0x0E),
		emu:read16(base + 0x10),
		emu:read16(base + 0x12), -- 7-10: moves
		emu:read8(base + 0x21),
		emu:read8(base + 0x22), -- 11-12: types
		emu:read8(base + 0x2C), -- 13: level
		emu:read32(base + 0x4C),
		emu:read32(base + 0x50), -- 14-15: status1, status2
		emu:read8(base + 0x19),
		emu:read8(base + 0x1A),
		emu:read8(base + 0x1B), -- 16-18: atk/def/spd buffs
		emu:read16(base + 0x00),
		emu:read16(base + 0x2E),
		emu:read8(base + 0x20), -- 19-21: species, item, ability
	}
end

function isPartyWiped()
	for i = 0, 5 do
		if emu:read16(PARTY_HP_BASE + (i * PARTY_STRIDE)) > 0 then
			return false
		end
	end
	return true
end

function getState()
	local b0 = getBattler(0x02024084) -- Player 1 (Left)
	local b1 = getBattler(0x020240DC) -- Enemy 1 (Left)
	local b2 = getBattler(0x02024134) -- Player 2 (Right)
	local b3 = getBattler(0x0202418C) -- Enemy 2 (Right)

	local battleOver, playerLost = 0, 0
	-- Simple battle over logic (b1 and b3 HP check)
	if b1[1] <= 0 and b3[1] <= 0 then
		battleOver = 1
	elseif b0[1] <= 0 and b2[1] <= 0 and isPartyWiped() then
		battleOver, playerLost = 1, 1
	end

	local p_party = {}
	local e_party = {}
	for i = 0, 5 do
		local p = readPartySlot(i) or EMPTY_SLOT
		local e = readEnemyPartySlot(i) or EMPTY_ENEMY_SLOT
		local php = emu:read16(PARTY_HP_BASE + (i * PARTY_STRIDE))
		local pmax = emu:read16(PARTY_HP_BASE + (i * PARTY_STRIDE) + 2)
		table.insert(p_party, php)
		table.insert(p_party, pmax)
		table.insert(p_party, p.species)
		table.insert(p_party, p.item)
		for m = 1, 4 do
			table.insert(p_party, p.moves[m])
		end

		table.insert(e_party, e.species)
		table.insert(e_party, e.item)
		table.insert(e_party, e.hp)
		table.insert(e_party, e.maxhp)
		for m = 1, 4 do
			table.insert(e_party, e.moves[m])
		end
	end

	local weather = emu:read8(0x02024082)

	local out = {}
	for _, v in ipairs(b0) do
		table.insert(out, v)
	end
	for _, v in ipairs(b1) do
		table.insert(out, v)
	end
	for _, v in ipairs(b2) do
		table.insert(out, v)
	end
	for _, v in ipairs(b3) do
		table.insert(out, v)
	end
	table.insert(out, battleOver)
	table.insert(out, playerLost)
	for _, v in ipairs(p_party) do
		table.insert(out, v)
	end
	for _, v in ipairs(e_party) do
		table.insert(out, v)
	end
	table.insert(out, weather)

	return table.concat(out, ",")
end

function checkToggleKey()
	if toggleCooldown > 0 then
		toggleCooldown = toggleCooldown - 1
		return
	end
	local selectHeld = (emu:getKey(C.GBA_KEY.L) == 1)
	if selectHeld and not lastSelectState then
		autoWalkEnabled = not autoWalkEnabled
		console:log("Auto-walk: " .. (autoWalkEnabled and "ON" or "OFF"))
		if not autoWalkEnabled then
			emu:clearKeys(0xFFFF)
		end
		toggleCooldown = 20
	end
	lastSelectState = selectHeld
end

function processWalk()
	if not autoWalkEnabled then
		return
	end

	local enemyHP = emu:read16(ADDR_ENEMY_HP)
	if enemyHP > 0 and enemyHP < 999 then
		return
	end

	walkTimer = walkTimer + 1
	local dir = WALK_SEQUENCE[walkStep + 1]
	emu:addKey(dir)

	if walkTimer >= 15 then
		emu:clearKeys(0xFFFF)
		walkStep = (walkStep + 1) % 4
		walkTimer = 0
	end
end

function doAction(actionIndex)
	pendingSwitch = nil
	switchStep = 0
	pendingAction = actionIndex
	actionStep = 1
	actionTimer = 0
end

function processAction()
	if not pendingAction then
		return
	end
	actionTimer = actionTimer + 1
	local pressTime = 2

	-- Parse the 0-11 action into a Move (0-3) and a Target (0=Left, 1=Right, 2=Partner)
	local moveIdx = pendingAction % 4
	local targetId = math.floor(pendingAction / 4)

	if actionStep == 1 then
		emu:addKey(C.GBA_KEY.B)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 2
			actionTimer = 0
		end
	elseif actionStep == 2 then
		if actionTimer >= pressTime then
			actionStep = 3
			actionTimer = 0
		end
	elseif actionStep == 3 then
		emu:addKey(C.GBA_KEY.B)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 4
			actionTimer = 0
		end
	elseif actionStep == 4 then
		if actionTimer >= pressTime then
			actionStep = 5
			actionTimer = 0
		end
	elseif actionStep == 5 then
		emu:addKey(C.GBA_KEY.UP)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 6
			actionTimer = 0
		end
	elseif actionStep == 6 then
		if actionTimer >= pressTime then
			actionStep = 7
			actionTimer = 0
		end
	elseif actionStep == 7 then
		emu:addKey(C.GBA_KEY.LEFT)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 8
			actionTimer = 0
		end
	elseif actionStep == 8 then
		if actionTimer >= pressTime then
			actionStep = 9
			actionTimer = 0
		end
	elseif actionStep == 9 then
		emu:addKey(C.GBA_KEY.A)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 10
			actionTimer = 0
		end
	elseif actionStep == 10 then
		if actionTimer >= 15 then
			actionStep = 11
			actionTimer = 0
		end
	elseif actionStep == 11 then
		emu:addKey(C.GBA_KEY.UP)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 12
			actionTimer = 0
		end
	elseif actionStep == 12 then
		if actionTimer >= pressTime then
			actionStep = 13
			actionTimer = 0
		end
	elseif actionStep == 13 then
		emu:addKey(C.GBA_KEY.LEFT)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 14
			actionTimer = 0
		end
	elseif actionStep == 14 then
		if actionTimer >= pressTime then
			actionStep = 15
			actionTimer = 0
		end
	elseif actionStep == 15 then
		if moveIdx == 1 or moveIdx == 3 then
			emu:addKey(C.GBA_KEY.RIGHT)
		end
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 16
			actionTimer = 0
		end
	elseif actionStep == 16 then
		if actionTimer >= pressTime then
			actionStep = 17
			actionTimer = 0
		end
	elseif actionStep == 17 then
		if moveIdx == 2 or moveIdx == 3 then
			emu:addKey(C.GBA_KEY.DOWN)
		end
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 18
			actionTimer = 0
		end
	elseif actionStep == 18 then
		if actionTimer >= pressTime then
			actionStep = 19
			actionTimer = 0
		end
	elseif actionStep == 19 then
		emu:addKey(C.GBA_KEY.A)
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 20
			actionTimer = 0
		end
	-- --- NEW TARGETING LOGIC START ---
	elseif actionStep == 20 then
		if actionTimer >= 20 then -- Wait for target menu to open
			actionStep = 21
			actionTimer = 0
		end
	elseif actionStep == 21 then
		if targetId == 0 or targetId == 1 then
			emu:addKey(C.GBA_KEY.UP) -- Aim at enemies
		elseif targetId == 2 then
			emu:addKey(C.GBA_KEY.DOWN) -- Aim at partner
		end
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 22
			actionTimer = 0
		end
	elseif actionStep == 22 then
		if actionTimer >= pressTime then
			actionStep = 23
			actionTimer = 0
		end
	elseif actionStep == 23 then
		if targetId == 0 then
			emu:addKey(C.GBA_KEY.LEFT) -- Aim at Left Enemy
		elseif targetId == 1 then
			emu:addKey(C.GBA_KEY.RIGHT) -- Aim at Right Enemy
		end
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 24
			actionTimer = 0
		end
	elseif actionStep == 24 then
		if actionTimer >= pressTime then
			actionStep = 25
			actionTimer = 0
		end
	elseif actionStep == 25 then
		emu:addKey(C.GBA_KEY.A) -- Confirm Target!
		if actionTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			actionStep = 0
			pendingAction = nil
		end
	end
end

function doSwitch(targetSlot)
	pendingAction = nil
	actionStep = 0
	pendingSwitch = true
	switchTargetSlot = targetSlot
	switchStep = 1
	switchTimer = 0
end

function processSwitch()
	if not pendingSwitch then
		return
	end
	switchTimer = switchTimer + 1
	local pressTime = 2

	if switchStep == 1 then
		emu:addKey(C.GBA_KEY.B)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 2
			switchTimer = 0
		end
	elseif switchStep == 2 then
		if switchTimer >= pressTime then
			switchStep = 3
			switchTimer = 0
		end
	elseif switchStep == 3 then
		emu:addKey(C.GBA_KEY.B)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 4
			switchTimer = 0
		end
	elseif switchStep == 4 then
		if switchTimer >= pressTime then
			switchStep = 5
			switchTimer = 0
		end
	elseif switchStep == 5 then
		emu:addKey(C.GBA_KEY.UP)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 6
			switchTimer = 0
		end
	elseif switchStep == 6 then
		if switchTimer >= pressTime then
			switchStep = 7
			switchTimer = 0
		end
	elseif switchStep == 7 then
		emu:addKey(C.GBA_KEY.LEFT)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 8
			switchTimer = 0
		end
	elseif switchStep == 8 then
		if switchTimer >= pressTime then
			switchStep = 9
			switchTimer = 0
		end
	elseif switchStep == 9 then
		emu:addKey(C.GBA_KEY.DOWN)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 10
			switchTimer = 0
		end
	elseif switchStep == 10 then
		if switchTimer >= pressTime then
			switchStep = 11
			switchTimer = 0
		end
	elseif switchStep == 11 then
		emu:addKey(C.GBA_KEY.A)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 12
			switchTimer = 0
		end
	elseif switchStep == 12 then
		if switchTimer >= 30 then
			switchStep = 13
			switchTimer = 0
		end
	elseif switchStep == 13 then
		local currentSlot = emu:read8(PARTY_MENU_SLOT_ADDR)
		if currentSlot == switchTargetSlot then
			switchStep = 14
			switchTimer = 0
		elseif switchTimer >= 200 then
			console:log("WARNING: couldn't reach target slot " .. switchTargetSlot .. ", stuck at " .. currentSlot)
			pendingSwitch = nil
			switchStep = 0
		else
			if currentSlot < switchTargetSlot then
				emu:addKey(C.GBA_KEY.DOWN)
			else
				emu:addKey(C.GBA_KEY.UP)
			end
			if switchTimer >= pressTime then
				emu:clearKeys(0xFFFF)
				switchTimer = 0
			end
		end
	elseif switchStep == 14 then
		emu:addKey(C.GBA_KEY.A)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 15
			switchTimer = 0
		end
	elseif switchStep == 15 then
		if switchTimer >= 20 then
			switchStep = 16
			switchTimer = 0
		end
	elseif switchStep == 16 then
		emu:addKey(C.GBA_KEY.A)
		if switchTimer >= pressTime then
			emu:clearKeys(0xFFFF)
			switchStep = 0
			pendingSwitch = nil
		end
	end
end

function onFrame()
	checkToggleKey()
	processWalk()

	local enemyHP = emu:read16(ADDR_ENEMY_HP)
	local playerHP = emu:read16(ADDR_PLAYER_HP)
	local inBattle = (enemyHP > 0 and enemyHP < 999 and playerHP > 0)

	if not inBattle and (pendingAction or pendingSwitch) then
		console:log("Battle ended mid-action, aborting pending action/switch")
		pendingAction = nil
		pendingSwitch = nil
		actionStep = 0
		switchStep = 0
		emu:clearKeys(0xFFFF)
	end

	processAction()
	processSwitch()

	if not client then
		local newClient = server:accept()
		if newClient then
			client = newClient
			console:log("Python client connected")
		end
	else
		if client:hasdata() then
			local line = client:receive(1024)
			if line == nil or line == "" then
				console:log("Client disconnected")
				client = nil
				return
			end
			line = line:gsub("%s+$", "")
			if line == "GET_STATE" then
				client:send(getState() .. "\n")
			elseif string.match(line, "^ACTION:") then
				local action = tonumber(string.match(line, "ACTION:(%d+)"))
				if action < 12 then
					doAction(action)
				else
					doSwitch(action - 11)
				end
				client:send("ACK\n")
			end
		end
	end
end

callbacks:add("frame", onFrame)
startServer()
