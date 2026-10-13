-- battle_bridge.lua

local server = nil
local client = nil

local ADDR_PLAYER_HP = 0x020240AC
local ADDR_PLAYER_MAX_HP = 0x020240B0
local ADDR_ENEMY_HP = 0x02024104
local ADDR_ENEMY_MAX_HP = 0x02024108
local PP_ADDRS = { 0x020240A8, 0x020240A9, 0x020240AA, 0x020240AB }

local PARTY_MENU_SLOT_ADDR = 0x0203CED1

local pendingAction = nil
local actionStep = 0
local actionTimer = 0

local pendingSwitch = nil
local switchTargetSlot = 0
local switchStep = 0
local switchTimer = 0

local autoWalkEnabled = false
local walkStep = 0
local walkTimer = 0
local lastSelectState = false
local toggleCooldown = 0

local PARTY_HP_BASE = 0x02024542
local PARTY_START = 0x020244EC
local PARTY_STRIDE = 0x64

local WALK_SEQUENCE = {
	C.GBA_KEY.UP,
	C.GBA_KEY.RIGHT,
	C.GBA_KEY.DOWN,
	C.GBA_KEY.LEFT,
}

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

local EMPTY_SLOT = { species = 0, item = 0, moves = { 0, 0, 0, 0 } }

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

function getMovePP(moveIndex)
	return emu:read8(PP_ADDRS[moveIndex + 1])
end

function isPartyWiped()
	for i = 0, 5 do
		local hp = emu:read16(PARTY_HP_BASE + (i * PARTY_STRIDE))
		if hp > 0 then
			return false
		end
	end
	return true
end

function getState()
	local playerHP = emu:read16(ADDR_PLAYER_HP)
	local playerMaxHP = emu:read16(ADDR_PLAYER_MAX_HP)
	local enemyHP = emu:read16(ADDR_ENEMY_HP)
	local enemyMaxHP = emu:read16(ADDR_ENEMY_MAX_HP)

	local pp1 = getMovePP(0)
	local pp2 = getMovePP(1)
	local pp3 = getMovePP(2)
	local pp4 = getMovePP(3)

	local move1 = emu:read16(ADDR_PLAYER_HP - 28)
	local move2 = emu:read16(ADDR_PLAYER_HP - 26)
	local move3 = emu:read16(ADDR_PLAYER_HP - 24)
	local move4 = emu:read16(ADDR_PLAYER_HP - 22)

	local p_type1 = emu:read8(ADDR_PLAYER_HP - 7)
	local p_type2 = emu:read8(ADDR_PLAYER_HP - 6)
	local e_type1 = emu:read8(ADDR_ENEMY_HP - 7)
	local e_type2 = emu:read8(ADDR_ENEMY_HP - 6)

	local p_level = emu:read8(ADDR_PLAYER_HP + 2)
	local e_level = emu:read8(ADDR_ENEMY_HP + 2)

	local pStatus1 = emu:read32(0x020240D0)
	local eStatus1 = emu:read32(0x02024128)
	local pStatus2 = emu:read32(0x020240D4)
	local eStatus2 = emu:read32(0x0202412C)

	local p_atk_buff = emu:read8(ADDR_PLAYER_HP - 15)
	local p_def_buff = emu:read8(ADDR_PLAYER_HP - 14)
	local p_spd_buff = emu:read8(ADDR_PLAYER_HP - 13)
	local e_atk_buff = emu:read8(ADDR_ENEMY_HP - 15)
	local e_def_buff = emu:read8(ADDR_ENEMY_HP - 14)
	local e_spd_buff = emu:read8(ADDR_ENEMY_HP - 13)

	local party1_hp = emu:read16(PARTY_HP_BASE + (0 * PARTY_STRIDE))
	local party1_maxhp = emu:read16(PARTY_HP_BASE + (0 * PARTY_STRIDE) + 2)
	local party2_hp = emu:read16(PARTY_HP_BASE + (1 * PARTY_STRIDE))
	local party2_maxhp = emu:read16(PARTY_HP_BASE + (1 * PARTY_STRIDE) + 2)
	local party3_hp = emu:read16(PARTY_HP_BASE + (2 * PARTY_STRIDE))
	local party3_maxhp = emu:read16(PARTY_HP_BASE + (2 * PARTY_STRIDE) + 2)
	local party4_hp = emu:read16(PARTY_HP_BASE + (3 * PARTY_STRIDE))
	local party4_maxhp = emu:read16(PARTY_HP_BASE + (3 * PARTY_STRIDE) + 2)
	local party5_hp = emu:read16(PARTY_HP_BASE + (4 * PARTY_STRIDE))
	local party5_maxhp = emu:read16(PARTY_HP_BASE + (4 * PARTY_STRIDE) + 2)
	local party6_hp = emu:read16(PARTY_HP_BASE + (5 * PARTY_STRIDE))
	local party6_maxhp = emu:read16(PARTY_HP_BASE + (5 * PARTY_STRIDE) + 2)

	local p1 = readPartySlot(0) or EMPTY_SLOT
	local p2 = readPartySlot(1) or EMPTY_SLOT
	local p3 = readPartySlot(2) or EMPTY_SLOT
	local p4 = readPartySlot(3) or EMPTY_SLOT
	local p5 = readPartySlot(4) or EMPTY_SLOT
	local p6 = readPartySlot(5) or EMPTY_SLOT

	local battleOver = 0
	local playerLost = 0

	if enemyHP <= 0 then
		battleOver = 1
	elseif playerHP <= 0 and isPartyWiped() then
		battleOver = 1
		playerLost = 1
	end

	return string.format(
		"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,"
			.. "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
		playerHP,
		playerMaxHP,
		enemyHP,
		enemyMaxHP,
		battleOver,
		playerLost,
		pp1,
		pp2,
		pp3,
		pp4,
		move1,
		move2,
		move3,
		move4,
		p_type1,
		p_type2,
		e_type1,
		e_type2,
		p_level,
		e_level,
		pStatus1,
		eStatus1,
		pStatus2,
		eStatus2,
		p_atk_buff,
		p_def_buff,
		p_spd_buff,
		e_atk_buff,
		e_def_buff,
		e_spd_buff,
		party1_hp,
		party1_maxhp,
		party2_hp,
		party2_maxhp,
		party3_hp,
		party3_maxhp,
		party4_hp,
		party4_maxhp,
		party5_hp,
		party5_maxhp,
		party6_hp,
		party6_maxhp,
		p1.species,
		p1.item,
		p1.moves[1],
		p1.moves[2],
		p1.moves[3],
		p1.moves[4],
		p2.species,
		p2.item,
		p2.moves[1],
		p2.moves[2],
		p2.moves[3],
		p2.moves[4],
		p3.species,
		p3.item,
		p3.moves[1],
		p3.moves[2],
		p3.moves[3],
		p3.moves[4],
		p4.species,
		p4.item,
		p4.moves[1],
		p4.moves[2],
		p4.moves[3],
		p4.moves[4],
		p5.species,
		p5.item,
		p5.moves[1],
		p5.moves[2],
		p5.moves[3],
		p5.moves[4],
		p6.species,
		p6.item,
		p6.moves[1],
		p6.moves[2],
		p6.moves[3],
		p6.moves[4]
	)
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
		if pendingAction == 1 or pendingAction == 3 then
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
		if pendingAction == 2 or pendingAction == 3 then
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
			switchStep = 14
			switchTimer = 0
		else
			emu:addKey(C.GBA_KEY.DOWN)
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
				if action < 4 then
					doAction(action)
				else
					doSwitch(action - 4)
				end
				client:send("ACK\n")
			end
		end
	end
end

callbacks:add("frame", onFrame)
startServer()
