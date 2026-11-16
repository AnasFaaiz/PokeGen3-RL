return function(D)
local CONFIG = D.CONFIG -- use the global `emu` (re-resolved each call) so a core reload cannot leave a stale handle
local partySlotBase, battleOutcome, readCb2, controllerFn, sameFn, block_orders = D.partySlotBase, D.battleOutcome, D.readCb2, D.controllerFn, D.sameFn, D.block_orders
local function validEwram(p)
	return p >= 0x02000000 and p < 0x02040000
end
local function inventoryLine()
	local I = CONFIG.INV
	local sb1 = emu:read32(CONFIG.OW.SAVEBLOCK1_PTR)
	local sb2 = emu:read32(I.SAVEBLOCK2_PTR)
	if not (validEwram(sb1) and validEwram(sb2)) then
		return "INV:ERR:bad_saveblock_ptr:" .. string.format("%08X,%08X", sb1, sb2)
	end
	local key = emu:read32(sb2 + I.SB2_ENC_KEY)
	local money = emu:read32(sb1 + I.SB1_MONEY) ~ key
	local balls = {}
	for i = 0, I.BALL_SLOTS - 1 do
		local a = sb1 + I.SB1_BALLS + i * 4
		local id = emu:read16(a)
		if id ~= 0 then
			balls[#balls + 1] = string.format("%d:%d", id, emu:read16(a + 2) ~ (key & 0xFFFF))
		end
	end
	local count = emu:read8(I.PARTY_COUNT_ADDR)
	local lv = {}
	for i = 0, math.min(count, 6) - 1 do
		lv[#lv + 1] = emu:read8(partySlotBase(i) + I.PARTY_LEVEL_OFFSET)
	end
	local btf = emu:read32(CONFIG.BATTLE_TYPE_FLAGS_ADDR)
	return string.format(
		"INV:money=%d;balls=%s;party=%d;lv=%s;outcome=%d;trainer=%d;map=%d.%d",
		money,
		table.concat(balls, ","),
		count,
		table.concat(lv, ","),
		battleOutcome(),
		(btf & I.BATTLE_TYPE_TRAINER_BIT) ~= 0 and 1 or 0,
		emu:read8(sb1 + 4),
		emu:read8(sb1 + 5)
	)
end
local function boxLine()
	local I = CONFIG.INV
	local st = emu:read32(I.STORAGE_PTR)
	if not validEwram(st) then
		return "BOX:ERR:bad_storage_ptr:" .. string.format("%08X", st)
	end
	local out = {}
	for b = 0, I.BOX_COUNT - 1 do
		for s = 0, I.BOX_SLOTS - 1 do
			local base = st + I.STORAGE_BOXES_OFFSET + (b * I.BOX_SLOTS + s) * I.BOX_MON_SIZE
			local pid = emu:read32(base)
			if pid ~= 0 then
				local key = pid ~ emu:read32(base + 4)
				local order = block_orders[(pid % 24) + 1]
				local g
				for i = 1, 4 do
					if order[i] == 0 then
						g = base + 0x20 + (i - 1) * 12
					end
				end
				local species = (emu:read32(g) ~ key) & 0xFFFF
				local exp = emu:read32(g + 4) ~ key
				if species ~= 0 then
					out[#out + 1] = string.format("%d.%d:%d:%d", b, s, species, exp)
				end
			end
		end
	end
	return "BOX:" .. table.concat(out, ",")
end
local function bagLine()
	local I = CONFIG.INV
	local bp = I.BAG_POSITION
	local pocket = emu:read8(bp + I.BAG_POCKET_OFFSET)
	local cursor, scroll = {}, {}
	for i = 0, 4 do
		cursor[#cursor + 1] = emu:read16(bp + I.BAG_CURSOR_OFFSET + i * 2)
		scroll[#scroll + 1] = emu:read16(bp + I.BAG_SCROLL_OFFSET + i * 2)
	end
	return string.format(
		"BAG:pocket=%d;cursor=%s;scroll=%s;cb2=%08X;ctrl0=%08X;inbag=%d;last_item=%d;outcome=%d",
		pocket,
		table.concat(cursor, ","),
		table.concat(scroll, ","),
		readCb2(),
		controllerFn(0),
		sameFn(readCb2(), I.CB2_BAG_RUN) and 1 or 0,
		emu:read16(I.LAST_USED_ITEM),
		battleOutcome()
	)
end
local function dexLine()
	local I = CONFIG.INV
	local sb2 = emu:read32(I.SAVEBLOCK2_PTR)
	if not validEwram(sb2) then
		return "DEX:ERR:bad_saveblock2_ptr"
	end
	local out = {}
	for i = 0, I.DEX_BYTES - 1 do
		out[#out + 1] = string.format("%02X", emu:read8(sb2 + I.SB2_DEX_OWNED + i))
	end
	return "DEX:" .. table.concat(out)
end
local function ballSlots()
	local I = CONFIG.INV
	local sb1 = emu:read32(CONFIG.OW.SAVEBLOCK1_PTR)
	local sb2 = emu:read32(I.SAVEBLOCK2_PTR)
	local slots, total = {}, 0
	if not (validEwram(sb1) and validEwram(sb2)) then
		return slots, total
	end
	local key = emu:read32(sb2 + I.SB2_ENC_KEY) & 0xFFFF
	for i = 0, I.BALL_SLOTS - 1 do
		local a = sb1 + I.SB1_BALLS + i * 4
		local id = emu:read16(a)
		if id ~= 0 then
			local qty = emu:read16(a + 2) ~ key
			slots[#slots + 1] = { id = id, qty = qty }
			total = total + qty
		end
	end
	return slots, total
end
return {validEwram = validEwram, inventoryLine = inventoryLine, boxLine = boxLine, dexLine = dexLine, bagLine = bagLine, ballSlots = ballSlots}
end
