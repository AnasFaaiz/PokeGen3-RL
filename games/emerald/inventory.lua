return function(D)
local CONFIG = D.CONFIG -- use the global `emu` (re-resolved each call) so a core reload cannot leave a stale handle
local partySlotBase, battleOutcome, readCb2, controllerFn, sameFn, block_orders = D.partySlotBase, D.battleOutcome, D.readCb2, D.controllerFn, D.sameFn, D.block_orders
local isInBattle = D.isInBattle
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
	-- species:exp per party slot (decrypted like a box mon), so the trainer can pay for
	-- fractional level progress instead of only whole level-ups.
	local px = {}
	for i = 0, math.min(count, 6) - 1 do
		local base = partySlotBase(i)
		local pid = emu:read32(base)
		if pid == 0 then
			px[#px + 1] = "0:0"   -- empty/unreadable slot: the trainer keeps its whole level
		else
			local pkey = pid ~ emu:read32(base + 4)
			local order = block_orders[(pid % 24) + 1]
			local g = base + 0x20
			for j = 1, 4 do
				if order[j] == 0 then g = base + 0x20 + (j - 1) * 12 end
			end
			px[#px + 1] = string.format("%d:%d", (emu:read32(g) ~ pkey) & 0xFFFF, emu:read32(g + 4) ~ pkey)
		end
	end
	local btf = emu:read32(CONFIG.BATTLE_TYPE_FLAGS_ADDR)
	return string.format(
		"INV:money=%d;balls=%s;party=%d;lv=%s;outcome=%d;trainer=%d;map=%d.%d;px=%s",
		money,
		table.concat(balls, ","),
		count,
		table.concat(lv, ","),
		battleOutcome(),
		(btf & I.BATTLE_TYPE_TRAINER_BIT) ~= 0 and 1 or 0,
		emu:read8(sb1 + 4),
		emu:read8(sb1 + 5),
		table.concat(px, ",")
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
				local g, m
				for i = 1, 4 do
					if order[i] == 0 then
						g = base + 0x20 + (i - 1) * 12
					elseif order[i] == 3 then
						m = base + 0x20 + (i - 1) * 12
					end
				end
				local species = (emu:read32(g) ~ key) & 0xFFFF
				local exp = emu:read32(g + 4) ~ key
				local isEgg = (((emu:read32(m + 4) ~ key) >> 30) & 1) == 1 -- eggs are useless in the party: never offer them
				if species ~= 0 and not isEgg then
					out[#out + 1] = string.format("%d.%d:%d:%d", b, s, species, exp)
				end
			end
		end
	end
	return "BOX:" .. table.concat(out, ",")
end

-- Address of the box mon `b.s` in PC storage, or nil + error text.
local function boxMonAddr(spec)
	local I = CONFIG.INV
	local b, s = string.match(spec or "", "^(%d+)%.(%d+)$")
	b, s = tonumber(b), tonumber(s)
	if not b or b >= I.BOX_COUNT or s >= I.BOX_SLOTS then
		return nil, "bad_box_slot"
	end
	local st = emu:read32(I.STORAGE_PTR)
	if not validEwram(st) then
		return nil, "bad_storage_ptr"
	end
	return st + I.STORAGE_BOXES_OFFSET + (b * I.BOX_SLOTS + s) * I.BOX_MON_SIZE
end
-- BOXMON:<pid>:<species>:<exp>:<evs hp,atk,def,spe,spa,spd>:<iv word>, from a mon's first 80 bytes.
local function monLine(base)
	local pid = emu:read32(base)
	if pid == 0 then
		return "BOXMON:ERR:empty"
	end
	local key = pid ~ emu:read32(base + 4)
	local order = block_orders[(pid % 24) + 1]
	local blk = {}
	for i = 1, 4 do
		blk[order[i]] = base + 0x20 + (i - 1) * 12
	end
	local species = (emu:read32(blk[0]) ~ key) & 0xFFFF
	local exp = emu:read32(blk[0] + 4) ~ key
	local e0, e1 = emu:read32(blk[2]) ~ key, (emu:read32(blk[2] + 4) & 0xFFFF) ~ (key & 0xFFFF)
	local evs = string.format("%d,%d,%d,%d,%d,%d", e0 & 255, (e0 >> 8) & 255, (e0 >> 16) & 255, e0 >> 24, e1 & 255, (e1 >> 8) & 255)
	local iv = emu:read32(blk[3] + 4) ~ key
	return string.format("BOXMON:%d:%d:%d:%s:%d", pid, species, exp, evs, iv)
end
local function boxMonLine(spec)
	local base, err = boxMonAddr(spec)
	if not base then
		return "BOXMON:ERR:" .. err
	end
	return monLine(base)
end
local function readMon(base)
	local t = {}
	for i = 0, 19 do
		t[i] = emu:read32(base + i * 4)
	end
	return t
end
local function writeMon(base, t)
	for i = 0, 19 do
		emu:write32(base + i * 4, t[i])
	end
end
-- SWAP:<party slot>:<b.s>:<level>:<maxhp>:<atk>:<def>:<spe>:<spa>:<spd>
-- Exchanges the 80-byte mon structs verbatim (checksum stays valid) and rewrites the
-- unencrypted party tail for the incoming mon at full HP. Overworld only.
local function handleSwap(line)
	local I = CONFIG.INV
	if isInBattle() then
		return "SWAP:ERR:in_battle"
	end
	local a = {}
	for tok in string.gmatch(line, "[^:]+") do
		a[#a + 1] = tok
	end
	local slot, spec = tonumber(a[2]), a[3]
	local level, hp = tonumber(a[4]), tonumber(a[5])
	local count = emu:read8(I.PARTY_COUNT_ADDR)
	if not (slot and level and hp and slot >= 0 and slot < count and level >= 1 and level <= 100 and #a == 10) then
		return "SWAP:ERR:bad_args"
	end
	local bbase, err = boxMonAddr(spec)
	if not bbase then
		return "SWAP:ERR:" .. err
	end
	local pbase = partySlotBase(slot)
	local incoming, outgoing = readMon(bbase), readMon(pbase)
	if incoming[0] == 0 then
		return "SWAP:ERR:empty_box_slot"
	end
	writeMon(bbase, outgoing)
	writeMon(pbase, incoming)
	emu:write32(pbase + 0x50, 0) -- status
	emu:write8(pbase + I.PARTY_LEVEL_OFFSET, level)
	emu:write8(pbase + 0x55, 0xFF) -- mail
	emu:write16(pbase + 0x56, hp) -- current HP
	emu:write16(pbase + 0x58, hp) -- max HP
	for i = 0, 4 do
		emu:write16(pbase + 0x5A + i * 2, tonumber(a[6 + i]) or 1) -- atk, def, spe, spa, spd
	end
	return "SWAP:OK"
end
-- SPECIES:<id> -> SPECIES:<id>:<growth>:<hp>:<atk>:<def>:<spe>:<spa>:<spd>:<type1>:<type2>
-- Straight from the ROM's species table, so level-from-exp and stats follow this ROM, not a reference file.
local function speciesLine(id)
	local I = CONFIG.INV
	if not id or id < 1 or id >= I.SPECIES_COUNT then
		return "SPECIES:ERR:bad_species"
	end
	local a = I.SPECIES_INFO_ADDR + id * I.SPECIES_INFO_SIZE
	local growth = emu:read8(a + I.SPECIES_GROWTH_OFFSET)
	local f = {}
	for i = 0, 5 do
		f[#f + 1] = emu:read8(a + i)
	end
	if growth > 5 or f[1] == 0 then
		return "SPECIES:ERR:implausible_entry:" .. growth .. ":" .. f[1]
	end
	return string.format("SPECIES:%d:%d:%s:%d:%d", id, growth, table.concat(f, ":"), emu:read8(a + 6), emu:read8(a + 7))
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
-- TRAINER -> TRAINER:<opponent trainer id>, or TRAINER:ERR:<why>. Needs INV.TRAINER_OPPONENT_ADDR (unverified, nil
-- until found live), so until then it reports no_address and the trainer id stays unknown.
local function trainerLine()
	local addr = CONFIG.INV.TRAINER_OPPONENT_ADDR
	if not addr then
		return "TRAINER:ERR:no_address"
	end
	return string.format("TRAINER:%d", emu:read16(addr))
end

-- TRAINERSCAN:<value> -> TRAINERSCAN:<addr>,<addr>,... (aligned EWRAM u16 cells equal to value, at most 40).
-- Read-only helper to find TRAINER_OPPONENT_ADDR: run it during a trainer fight with that trainer's id,
-- then again during a different trainer's fight, and keep the address that matched both times.
local function trainerScanLine(value)
	if not value or value < 0 or value > 0xFFFF then
		return "TRAINERSCAN:ERR:bad_value"
	end
	local hits = {}
	for a = 0x02000000, 0x0203FFFE, 2 do
		if emu:read16(a) == value then
			hits[#hits + 1] = string.format("0x%08X", a)
			if #hits >= 40 then
				break
			end
		end
	end
	return "TRAINERSCAN:" .. table.concat(hits, ",")
end

-- TRAINERFLAGS:<id> -> TRAINERFLAGS:<id>:<0|1> (1 = already defeated). UNVERIFIED flag layout (see config).
local function trainerFlagLine(id)
	local I = CONFIG.INV
	local sb1 = emu:read32(CONFIG.OW.SAVEBLOCK1_PTR)
	if not validEwram(sb1) then
		return "TRAINERFLAGS:ERR:bad_saveblock_ptr"
	end
	if not id or id < 1 or id >= I.MAX_TRAINERS_COUNT then
		return "TRAINERFLAGS:ERR:bad_id"
	end
	local flag = I.TRAINER_FLAGS_START + id
	local byte = emu:read8(sb1 + I.SB1_FLAGS_OFFSET + (flag >> 3))
	return string.format("TRAINERFLAGS:%d:%d", id, (byte >> (flag & 7)) & 1)
end
-- ---------------------------------------------------------------- move teaching (UNVERIFIED live)
-- A mon's moves live in its Attacks substruct (moves u16 x4, PP u8 x4), encrypted with key = pid ^ otId and covered by a
-- checksum at +0x1C (sum of the 24 u16 words of the decrypted 48 data bytes). This mirrors training/mon_struct.py.
local function monBlocks(base)
	local pid = emu:read32(base)
	if pid == 0 then
		return nil
	end
	local key = pid ~ emu:read32(base + 4)
	local order = block_orders[(pid % 24) + 1]
	local addr, words = {}, {}
	for i = 1, 4 do
		local t = order[i]
		local a = base + 0x20 + (i - 1) * 12
		addr[t] = a
		words[t] = { emu:read32(a) ~ key, emu:read32(a + 4) ~ key, emu:read32(a + 8) ~ key }
	end
	return pid, key, addr, words
end
local function blocksChecksum(words)
	local sum = 0
	for t = 0, 3 do
		for j = 1, 3 do
			local w = words[t][j]
			sum = sum + (w & 0xFFFF) + (w >> 16)
		end
	end
	return sum & 0xFFFF
end
-- MONMOVES:<slot> -> MONMOVES:<pid>:<species>:<level>:<m1>,<m2>,<m3>,<m4>:<pp1>,..,<pp4>:<ppBonuses>:<checksum_ok 0|1>
local function monMovesLine(slot)
	local I = CONFIG.INV
	local count = emu:read8(I.PARTY_COUNT_ADDR)
	if not slot or slot < 0 or slot >= count then
		return "MONMOVES:ERR:bad_slot"
	end
	local base = partySlotBase(slot)
	local pid, _, _, w = monBlocks(base)
	if not pid then
		return "MONMOVES:ERR:empty"
	end
	local g, a = w[0], w[1]
	local ok = (emu:read16(base + 0x1C) == blocksChecksum(w)) and 1 or 0
	return string.format("MONMOVES:%d:%d:%d:%d,%d,%d,%d:%d,%d,%d,%d:%d:%d", pid, g[1] & 0xFFFF,
		emu:read8(base + I.PARTY_LEVEL_OFFSET), a[1] & 0xFFFF, a[1] >> 16, a[2] & 0xFFFF, a[2] >> 16,
		a[3] & 255, (a[3] >> 8) & 255, (a[3] >> 16) & 255, a[3] >> 24, g[3] & 255, ok)
end
-- SETMOVES:<slot>:<pid>:<m1>:<m2>:<m3>:<m4>:<pp1>:<pp2>:<pp3>:<pp4>[:<ppBonuses>][:dry]
-- The optional ppBonuses byte replaces the Growth substruct's PP-Up bits (the game clears a slot's bits when a move there
-- is replaced); without it they are left alone.
-- Overworld only. Refuses when the pid at that slot differs (the party was reordered), when the stored checksum was
-- already wrong, or when any value is out of range. After writing it re-reads the struct and puts the original words
-- back if anything does not match. "dry" runs every check and writes nothing.
local function setMovesLine(line)
	local I = CONFIG.INV
	if isInBattle() then
		return "SETMOVES:ERR:in_battle"
	end
	local t = {}
	for tok in string.gmatch(line, "[^:]+") do
		t[#t + 1] = tok
	end
	local dry = t[#t] == "dry"
	local bonus = tonumber(t[12])
	local slot, pid = tonumber(t[2]), tonumber(t[3])
	local mv, pp = {}, {}
	for i = 1, 4 do
		mv[i], pp[i] = tonumber(t[3 + i]), tonumber(t[7 + i])
	end
	if not (slot and pid and #t >= 11) then
		return "SETMOVES:ERR:bad_args"
	end
	for i = 1, 4 do
		if not (mv[i] and pp[i] and mv[i] >= 0 and mv[i] < 355 and pp[i] >= 0 and pp[i] <= 255) then
			return "SETMOVES:ERR:bad_args"
		end
	end
	local count = emu:read8(I.PARTY_COUNT_ADDR)
	if slot < 0 or slot >= count then
		return "SETMOVES:ERR:bad_slot"
	end
	local base = partySlotBase(slot)
	local cur, key, addr, w = monBlocks(base)
	if not cur then
		return "SETMOVES:ERR:empty"
	end
	if cur ~= pid then
		return "SETMOVES:ERR:pid_mismatch"
	end
	if emu:read16(base + 0x1C) ~= blocksChecksum(w) then
		return "SETMOVES:ERR:bad_checksum"
	end
	if bonus and (bonus < 0 or bonus > 255) then
		return "SETMOVES:ERR:bad_args"
	end
	local old = { w[1][1], w[1][2], w[1][3] }
	local oldGrowth3 = w[0][3]
	local oldSum = emu:read16(base + 0x1C)
	local a = { mv[1] | (mv[2] << 16), mv[3] | (mv[4] << 16), pp[1] | (pp[2] << 8) | (pp[3] << 16) | (pp[4] << 24) }
	w[1] = a
	if bonus then
		w[0][3] = (oldGrowth3 & ~0xFF) | bonus
	end
	local newSum = blocksChecksum(w)
	if dry then
		return "SETMOVES:DRY:ok"
	end
	for j = 1, 3 do
		emu:write32(addr[1] + (j - 1) * 4, a[j] ~ key)
	end
	if bonus then
		emu:write32(addr[0] + 8, w[0][3] ~ key)
	end
	emu:write16(base + 0x1C, newSum)
	local _, _, _, w2 = monBlocks(base)
	local good = w2 and emu:read16(base + 0x1C) == blocksChecksum(w2)
	for j = 1, 3 do
		good = good and w2[1][j] == a[j]
	end
	good = good and w2[0][3] == w[0][3]
	if not good then
		for j = 1, 3 do
			emu:write32(addr[1] + (j - 1) * 4, old[j] ~ key)
		end
		emu:write32(addr[0] + 8, oldGrowth3 ~ key)
		emu:write16(base + 0x1C, oldSum)
		return "SETMOVES:ERR:verify_failed"
	end
	return "SETMOVES:OK"
end
return {monMovesLine = monMovesLine, setMovesLine = setMovesLine, trainerLine = trainerLine, trainerScanLine = trainerScanLine, trainerFlagLine = trainerFlagLine, validEwram = validEwram, inventoryLine = inventoryLine, boxLine = boxLine, boxMonLine = boxMonLine, speciesLine = speciesLine, partyMonLine = function(slot) return monLine(partySlotBase(slot)) end, handleSwap = handleSwap, dexLine = dexLine, bagLine = bagLine, ballSlots = ballSlots}
end
