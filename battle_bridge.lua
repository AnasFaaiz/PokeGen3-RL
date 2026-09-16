-- battle_bridge.lua

local server = nil
local client = nil

local ADDR_PLAYER_HP = 0x020240AC
local ADDR_PLAYER_MAX_HP = 0x020240B0
local ADDR_ENEMY_HP = 0x02024104
local ADDR_ENEMY_MAX_HP = 0x02024108

local pendingAction = nil
local actionStep = 0
local actionTimer = 0

local autoWalkEnabled = false
local walkStep = 0
local walkTimer = 0
local lastSelectState = false
local toggleCooldown = 0

local WALK_SEQUENCE = {
	C.GBA_KEY.UP,
	C.GBA_KEY.RIGHT,
	C.GBA_KEY.DOWN,
	C.GBA_KEY.LEFT,
}

function startServer()
	server = socket.bind(nil, 8888)
	server:listen()
	console:log("Lua server listening on port 8888")
end

function getState()
	local playerHP = emu:read16(ADDR_PLAYER_HP)
	local playerMaxHP = emu:read16(ADDR_PLAYER_MAX_HP)
	local enemyHP = emu:read16(ADDR_ENEMY_HP)
	local enemyMaxHP = emu:read16(ADDR_ENEMY_MAX_HP)
	local battleOver = 0
	if enemyHP <= 0 or playerHP <= 0 then
		battleOver = 1
	end
	return string.format("%d,%d,%d,%d,%d", playerHP, playerMaxHP, enemyHP, enemyMaxHP, battleOver)
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
	pendingAction = actionIndex
	actionStep = 1
	actionTimer = 0
end

function processAction()
	if not pendingAction then
		return
	end
	actionTimer = actionTimer + 1

	if actionStep == 1 then
		emu:addKey(C.GBA_KEY.A)
		if actionTimer >= 2 then
			emu:clearKeys(0xFFFF)
			actionStep = 2
			actionTimer = 0
		end
	elseif actionStep == 2 then
		if pendingAction == 1 or pendingAction == 3 then
			emu:addKey(C.GBA_KEY.RIGHT)
		end
		if actionTimer >= 2 then
			emu:clearKeys(0xFFFF)
			actionStep = 3
			actionTimer = 0
		end
	elseif actionStep == 3 then
		if pendingAction == 2 or pendingAction == 3 then
			emu:addKey(C.GBA_KEY.DOWN)
		end
		if actionTimer >= 2 then
			emu:clearKeys(0xFFFF)
			actionStep = 4
			actionTimer = 0
		end
	elseif actionStep == 4 then
		emu:addKey(C.GBA_KEY.A)
		if actionTimer >= 2 then
			emu:clearKeys(0xFFFF)
			actionStep = 0
			pendingAction = nil
		end
	end
end

function onFrame()
	checkToggleKey()
	processWalk()
	processAction()

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
				doAction(action)
				client:send("ACK\n")
			end
		end
	end
end

callbacks:add("frame", onFrame)
startServer()
