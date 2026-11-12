-- overworld_nav.lua -- scripted overworld navigation for the RL harness.
--
-- The model only ever plays battles. Getting back to wild grass after a
-- battle, and walking to a Pokemon Center to heal before the party faints,
-- are handled here as a script so none of it leaks into the policy's
-- observation or action space.
--
-- Everything is driven by the game's own map data rather than by timing or
-- by pixels:
--   * gBackupMapLayout is a width*height array of u16, one per tile, packing
--     metatile id (bits 0-9), collision (10-11) and elevation (12-15).
--   * A tile's "behavior" (grass / door / cliff edge / ...) comes from the
--     map's tilesets, looked up by metatile id.
--   * Passability replicates GetCollisionAtCoords from the decomp: collision
--     bits, elevation compatibility, directional walls, and object events.
--     Collision bits alone are NOT enough -- cliff edges are walkable tiles
--     that may only be entered or left from certain sides, and planning
--     without that check strands the player against them.
--
-- Steps are confirmed by watching the player's coordinates change, never by
-- waiting a fixed number of frames.
--
-- This module owns no emulator globals; the bridge injects them via init().

local Nav = {}

local C, OW -- config tables
local mem -- { read8, read16, read32, readRange }
local io_ -- { hold(key), release() }
local KEY

local MAP_OFFSET = 7 -- the current map starts at (7,7) in the backup grid

-- Tunables (frames @ 60fps)
local STEP_TURN_FRAMES = 50 -- turning then walking a tile; blocked after this
local STEP_MAX_FRAMES = 90
local TASK_MAX_FRAMES = 60 * 90
local TALK_TAP_PERIOD = 14
local MAX_REPLANS = 40
local WANDER_MAX_FRAMES = 60 * 60
-- How far the player may stray from the grass tile it last stood on before
-- that patch is given up and a fresh one is looked for.
local WANDER_ANCHOR_RANGE = 8

-- ============================================================ module state

local S = nil -- current task, or nil
local behaviorCache = {}
local behaviorCacheKey = nil
local mapChain = {} -- maps walked through, oldest first: { {g=,n=}, ... }
-- Map where we last stepped around looking for encounters (grass or cave).
-- Survives a HEAL so the trip back returns here, not to the nearest grass.
local huntMap = nil
-- Steps the game refused although canStep() allowed them, keyed "x,y>dir".
-- Two failures wall the edge off for the rest of this map visit, so the
-- planner routes around whatever rule canStep() does not mirror instead of
-- pressing into it forever.
local blockedEdges, blockedCounts = {}, {}

function Nav.init(config, memApi, ioApi, keys)
	C, OW, mem, io_, KEY = config, config.OW, memApi, ioApi, keys
end

-- ============================================================ memory reads

local function s16(v)
	return (v >= 0x8000) and (v - 0x10000) or v
end

local function playerObjEvent()
	local id = mem.read8(OW.PLAYER_AVATAR + OW.PA_OBJECT_EVENT_ID)
	if id >= OW.OBJECT_EVENT_COUNT then
		return nil
	end
	return OW.OBJECT_EVENTS + id * OW.OBJECT_EVENT_SIZE
end

function Nav.playerPos()
	local oe = playerObjEvent()
	if not oe then
		return nil
	end
	return s16(mem.read16(oe + OW.OE_COORDS)), s16(mem.read16(oe + OW.OE_COORDS + 2))
end

local function playerElevation()
	local oe = playerObjEvent()
	if not oe then
		return 3
	end
	return mem.read8(oe + OW.OE_ELEVATION) & 0x0F
end

local function playerSettled()
	-- tileTransitionState 0 means the player is sitting on a tile, not
	-- part-way between two.
	return mem.read8(OW.PLAYER_AVATAR + OW.PA_TILE_TRANSITION) == 0
end

-- True while the game has taken input away from the player: a Match Call, an
-- NPC or sign's text box, an item pickup, and so on. All of these freeze
-- walking the same way, so the nav script doesn't need to tell them apart --
-- it just needs to notice input is locked and wait it out instead of reading
-- every blocked step as a wall it bumped into.
local function fieldLocked()
	return mem.read8(OW.LOCK_FIELD_CONTROLS) ~= 0
end

local function log(msg)
	if io_ and io_.log then
		io_.log("[nav] " .. msg)
	end
end

local function currentMap()
	local sb1 = mem.read32(OW.SAVEBLOCK1_PTR)
	if sb1 == 0 then
		return nil
	end
	local base = sb1 + OW.SB1_LOCATION
	return mem.read8(base + OW.WD_MAP_GROUP), mem.read8(base + OW.WD_MAP_NUM)
end

local function healLocation()
	local sb1 = mem.read32(OW.SAVEBLOCK1_PTR)
	if sb1 == 0 then
		return nil
	end
	local base = sb1 + OW.SB1_LAST_HEAL_LOCATION
	return mem.read8(base + OW.WD_MAP_GROUP),
		mem.read8(base + OW.WD_MAP_NUM),
		s16(mem.read16(base + OW.WD_X)),
		s16(mem.read16(base + OW.WD_Y))
end

-- Party HP straight from gPlayerParty. hp and maxHP sit outside the
-- encrypted substructures, so no decryption is needed here.
-- Returns current HP, max HP, and how many Pokemon can still fight. The
-- conscious count matters on its own: a team can sit above the HP threshold
-- while being one Pokemon away from a wipe.
function Nav.partyHp()
	local cur, max, conscious = 0, 0, 0
	for slot = 0, 5 do
		local base = C.PARTY_START + slot * C.PARTY_STRIDE
		local mx = mem.read16(base + C.PARTY_MAXHP_OFFSET)
		if mx > 0 and mx < 1000 then
			local hp = mem.read16(base + C.PARTY_HP_OFFSET)
			cur = cur + hp
			max = max + mx
			if hp > 0 then
				conscious = conscious + 1
			end
		end
	end
	return cur, max, conscious
end

local function partyFullyHealed()
	local cur, max = Nav.partyHp()
	return max > 0 and cur == max
end

-- ============================================================= map reading

-- metatile id -> behavior byte, through the map's two tilesets. Cached per
-- map layout: the lookup is two pointer hops into ROM and BFS would
-- otherwise repeat it for every tile it touches.
local function buildBehaviorCache()
	local layout = mem.read32(OW.MAP_HEADER + OW.MH_MAP_LAYOUT)
	if layout == 0 then
		return false
	end
	if behaviorCacheKey == layout then
		return true
	end

	local primary = mem.read32(layout + OW.ML_PRIMARY_TILESET)
	local secondary = mem.read32(layout + OW.ML_SECONDARY_TILESET)
	local pAttr = (primary ~= 0) and mem.read32(primary + OW.TS_METATILE_ATTRIBUTES) or 0
	local sAttr = (secondary ~= 0) and mem.read32(secondary + OW.TS_METATILE_ATTRIBUTES) or 0

	local cache = {}
	for id = 0, OW.NUM_METATILES_TOTAL - 1 do
		local attrAddr
		if id < OW.NUM_METATILES_IN_PRIMARY then
			attrAddr = (pAttr ~= 0) and (pAttr + id * 2) or nil
		else
			attrAddr = (sAttr ~= 0) and (sAttr + (id - OW.NUM_METATILES_IN_PRIMARY) * 2) or nil
		end
		if attrAddr then
			cache[id] = mem.read16(attrAddr) & OW.METATILE_ATTR_BEHAVIOR_MASK
		else
			cache[id] = 0
		end
	end
	behaviorCache, behaviorCacheKey = cache, layout
	return true
end

-- Snapshot of the walkable grid. Read once per plan, not per frame.
local function readGrid()
	if not buildBehaviorCache() then
		return nil
	end
	local w = mem.read32(OW.BACKUP_MAP_LAYOUT)
	local h = mem.read32(OW.BACKUP_MAP_LAYOUT + 4)
	local ptr = mem.read32(OW.BACKUP_MAP_LAYOUT + 8)
	if w <= 0 or h <= 0 or w > 1000 or h > 1000 or ptr == 0 then
		return nil
	end

	local cells, n = {}, w * h
	local raw = mem.readRange and mem.readRange(ptr, n * 2) or nil
	if raw and #raw == n * 2 then
		for i = 0, n - 1 do
			cells[i] = string.unpack("<I2", raw, i * 2 + 1)
		end
	else
		for i = 0, n - 1 do
			cells[i] = mem.read16(ptr + i * 2)
		end
	end

	-- Tiles occupied by NPCs. They move, so this is only a snapshot; a step
	-- that fails because someone walked into the way just triggers a replan.
	local blockedByObject = {}
	for i = 0, OW.OBJECT_EVENT_COUNT - 1 do
		local base = OW.OBJECT_EVENTS + i * OW.OBJECT_EVENT_SIZE
		if (mem.read8(base + OW.OE_ACTIVE_BYTE) & 1) == 1 then
			local ox = s16(mem.read16(base + OW.OE_COORDS))
			local oy = s16(mem.read16(base + OW.OE_COORDS + 2))
			if ox >= 0 and oy >= 0 and ox < w and oy < h then
				blockedByObject[ox + oy * w] = true
			end
		end
	end
	local px, py = Nav.playerPos()
	if px then
		blockedByObject[px + py * w] = nil
	end

	return { w = w, h = h, cells = cells, objects = blockedByObject, elevation = playerElevation() }
end

local function cellAt(g, x, y)
	if x < 0 or y < 0 or x >= g.w or y >= g.h then
		return nil
	end
	return g.cells[x + y * g.w]
end

local function behaviorAt(g, x, y)
	local c = cellAt(g, x, y)
	if not c then
		return nil
	end
	return behaviorCache[c & OW.MAPGRID_METATILE_ID_MASK] or 0
end

-- Replicates GetCollisionAtCoords: collision bits, elevation, directional
-- walls (checked on both the tile left and the tile entered), object events.
local function canStep(g, fx, fy, tx, ty, dir)
	local to = cellAt(g, tx, ty)
	if to == nil then
		return false
	end
	if blockedEdges[fx .. "," .. fy .. ">" .. dir] then
		return false
	end
	if (to & OW.MAPGRID_METATILE_ID_MASK) == OW.MAPGRID_UNDEFINED then
		return false
	end
	if (to & OW.MAPGRID_COLLISION_MASK) ~= 0 then
		return false
	end
	if g.objects[tx + ty * g.w] then
		return false
	end

	local mapElev = (to & OW.MAPGRID_ELEVATION_MASK) >> OW.MAPGRID_ELEVATION_SHIFT
	local pElev = g.elevation
	if
		pElev ~= OW.ELEVATION_TRANSITION
		and mapElev ~= OW.ELEVATION_TRANSITION
		and mapElev ~= OW.ELEVATION_MULTI_LEVEL
		and mapElev ~= pElev
	then
		return false
	end

	local fromB = behaviorAt(g, fx, fy)
	local toB = behaviorAt(g, tx, ty)
	if OW.LEDGE_BEHAVIORS[toB] then
		return false
	end

	-- Leaving `from` outward, and entering `to` inward.
	if dir == OW.DIR_SOUTH then
		if OW.BLOCKED_SOUTH[fromB] or OW.BLOCKED_NORTH[toB] then
			return false
		end
	elseif dir == OW.DIR_NORTH then
		if OW.BLOCKED_NORTH[fromB] or OW.BLOCKED_SOUTH[toB] then
			return false
		end
	elseif dir == OW.DIR_WEST then
		if OW.BLOCKED_WEST[fromB] or OW.BLOCKED_EAST[toB] then
			return false
		end
	elseif dir == OW.DIR_EAST then
		if OW.BLOCKED_EAST[fromB] or OW.BLOCKED_WEST[toB] then
			return false
		end
	end
	return true
end

local DIRS = {
	{ d = 2, dx = 0, dy = -1 }, -- DIR_NORTH
	{ d = 1, dx = 0, dy = 1 }, -- DIR_SOUTH
	{ d = 3, dx = -1, dy = 0 }, -- DIR_WEST
	{ d = 4, dx = 1, dy = 0 }, -- DIR_EAST
}

local DIR_TO_KEY = nil -- built in init-on-first-use, needs KEY

local function dirKey(d)
	if not DIR_TO_KEY then
		DIR_TO_KEY = { [1] = KEY.DOWN, [2] = KEY.UP, [3] = KEY.LEFT, [4] = KEY.RIGHT }
	end
	return DIR_TO_KEY[d]
end

-- ==================================================================== BFS

-- Breadth-first search to the nearest tile satisfying goalFn. Returns a list
-- of directions, plus the goal coordinates.
--
-- `avoid` is an optional predicate for tiles to route around (used to keep a
-- weakened party out of grass on the way to heal). Callers run BFS twice:
-- once avoiding, then once without, so avoidance never makes a goal
-- unreachable. That is cheaper and simpler than a weighted search.
local function bfs(g, sx, sy, goalFn, avoid)
	local w, h = g.w, g.h
	local start = sx + sy * w
	local visited, parent, pdir = { [start] = true }, {}, {}
	local queue, qh, qt = { start }, 1, 1

	if goalFn(g, sx, sy) then
		return {}, sx, sy
	end

	while qh <= qt do
		local cur = queue[qh]
		qh = qh + 1
		local cx, cy = cur % w, cur // w
		for i = 1, 4 do
			local D = DIRS[i]
			local nx, ny = cx + D.dx, cy + D.dy
			local nidx = nx + ny * w
			if
				nx >= 0
				and ny >= 0
				and nx < w
				and ny < h
				and not visited[nidx]
				and canStep(g, cx, cy, nx, ny, D.d)
				and not (avoid and avoid(g, nx, ny))
			then
				visited[nidx] = true
				parent[nidx], pdir[nidx] = cur, D.d
				if goalFn(g, nx, ny) then
					local path, node = {}, nidx
					while node ~= start do
						table.insert(path, 1, pdir[node])
						node = parent[node]
					end
					return path, nx, ny
				end
				qt = qt + 1
				queue[qt] = nidx
			end
		end
	end
	return nil
end

local function bfsPreferAvoiding(g, sx, sy, goalFn, avoid)
	local p, gx, gy = bfs(g, sx, sy, goalFn, avoid)
	if p then
		return p, gx, gy
	end
	return bfs(g, sx, sy, goalFn, nil)
end

-- bfs() returns a shortest path, but reconstructs it as runs of one direction
-- at a time (e.g. 4 souths then 3 easts) rather than anything that looks like
-- a natural walk. That makes the player march in a straight line and take a
-- hard 90-degree turn right at the corner, instead of angling toward the
-- target -- and it can walk straight past a tile that's diagonally adjacent
-- rather than turning for it. Interleaving the two run-lengths proportionally
-- (south-east-south-east-south-east-south-south, not south*4-east*3) keeps
-- the same length and endpoint -- while producing a diagonal-looking walk
-- instead of an L-shape. It is NOT the same route: the reordered steps visit
-- different tiles, which may be cliffs or walls (see interleavePath below).
local function interleaveRaw(path)
	if not path or #path <= 2 then
		return path
	end

	local runs, order = {}, {}
	for _, d in ipairs(path) do
		if not runs[d] then
			runs[d] = 0
			order[#order + 1] = d
		end
		runs[d] = runs[d] + 1
	end
	if #order < 2 then
		return path
	end -- already a single direction

	-- Evenly spread each direction's steps across the full path length so
	-- no two directions' turns clump together (handles >2 distinct
	-- directions too, in case a future goal shape produces one).
	local out, pos = {}, {}
	for _, d in ipairs(order) do
		pos[d] = 0
	end
	local total = #path
	for i = 1, total do
		local best, bestFrac
		for _, d in ipairs(order) do
			if pos[d] < runs[d] then
				-- How far "behind schedule" this direction is at step i.
				local frac = pos[d] / runs[d]
				if not bestFrac or frac < bestFrac then
					best, bestFrac = d, frac
				end
			end
		end
		pos[best] = pos[best] + 1
		out[i] = best
	end
	return out
end

-- Interleaving changes which tiles are walked, and the bfs path only proved
-- ITS tiles walkable. Re-check every step of the reordered walk with canStep
-- from the real start tile and keep the original (known-good) path if any
-- step would be refused. Without this the player cut diagonals through cliffs.
local function interleavePath(path, g, x, y)
	local out = interleaveRaw(path)
	if out == path or not g or not x then
		return path
	end
	local cx, cy = x, y
	for _, d in ipairs(out) do
		local D
		for i = 1, 4 do
			if DIRS[i].d == d then
				D = DIRS[i]
			end
		end
		local nx, ny = cx + D.dx, cy + D.dy
		if not canStep(g, cx, cy, nx, ny, d) then
			return path
		end
		cx, cy = nx, ny
	end
	return out
end

-- ============================================================ goal helpers

local function isEncounter(g, x, y)
	return OW.ENCOUNTER_BEHAVIORS[behaviorAt(g, x, y)] == true
end

-- Warp tiles come from the map's warp event table, not from tile behaviour.
-- Behaviour misses most of them: a doorway is often an ordinary-looking tile
-- with a warp event sitting on it, so matching MB_ANIMATED_DOOR and friends
-- found none of Route 110's six warps. The event table is exact.
local warpTileSet, warpTileSetKey = {}, nil

local function refreshWarpTiles()
	local events = mem.read32(OW.MAP_HEADER + OW.MH_EVENTS)
	if warpTileSetKey == events then
		return
	end
	local set = {}
	for _, wp in ipairs(Nav.mapWarps()) do
		set[wp.x .. "," .. wp.y] = true
	end
	warpTileSet, warpTileSetKey = set, events
end

local function isWarpTile(g, x, y)
	if warpTileSet[x .. "," .. y] then
		return true
	end
	return OW.WARP_BEHAVIORS[behaviorAt(g, x, y)] == true
end

-- Warps declared by the current map, as {x, y, mapGroup, mapNum}. These are
-- map-grid coordinates (the event table stores map-local ones).
function Nav.mapWarps()
	local events = mem.read32(OW.MAP_HEADER + OW.MH_EVENTS)
	if events == 0 then
		return {}
	end
	local count = mem.read8(events + OW.ME_WARP_COUNT)
	local arr = mem.read32(events + OW.ME_WARPS)
	if arr == 0 or count == 0 or count > 64 then
		return {}
	end
	local out = {}
	for i = 0, count - 1 do
		local base = arr + i * OW.WARP_EVENT_SIZE
		out[#out + 1] = {
			x = s16(mem.read16(base + OW.WE_X)) + MAP_OFFSET,
			y = s16(mem.read16(base + OW.WE_Y)) + MAP_OFFSET,
			group = mem.read8(base + OW.WE_MAP_GROUP),
			num = mem.read8(base + OW.WE_MAP_NUM),
		}
	end
	return out
end

-- The warp on this map that leads to `group`/`num`, if any.
local function warpTo(group, num)
	for _, warp in ipairs(Nav.mapWarps()) do
		if warp.group == group and warp.num == num then
			return warp
		end
	end
	return nil
end

-- The Pokemon Center's front door on the heal map.
--
-- lastHealLocation is NOT inside the Center: for a city it is the outdoor
-- tile you respawn on, directly in front of the Center's door (Slateport's
-- is (19,20), with the door at (19,19)). So the door is the warp nearest
-- that tile. Scanning for "the nearest warp" rather than matching a map id
-- avoids needing to know which map id each Center has.
local function pokemonCenterDoor(hx, hy)
	local best, bestDist
	for _, warp in ipairs(Nav.mapWarps()) do
		local d = math.abs(warp.x - hx) + math.abs(warp.y - hy)
		if d <= 3 and (not bestDist or d < bestDist) then
			best, bestDist = warp, d
		end
	end
	return best
end

-- The healing counter, located by the nurse's sprite id. Returns her tile.
local function nurseTile()
	for i = 0, OW.OBJECT_EVENT_COUNT - 1 do
		local base = OW.OBJECT_EVENTS + i * OW.OBJECT_EVENT_SIZE
		if
			(mem.read8(base + OW.OE_ACTIVE_BYTE) & 1) == 1
			and mem.read8(base + OW.OE_GRAPHICS_ID) == OW.NURSE_GFX_ID
		then
			return s16(mem.read16(base + OW.OE_COORDS)), s16(mem.read16(base + OW.OE_COORDS + 2))
		end
	end
	return nil
end

-- ============================================================ map chain

local function sameMap(a, b)
	return a and b and a.g == b.g and a.n == b.n
end

local function noteCurrentMap()
	local g, n = currentMap()
	if not g then
		return
	end
	local cur = { g = g, n = n }
	local last = mapChain[#mapChain]
	if sameMap(last, cur) then
		return
	end
	blockedEdges, blockedCounts = {}, {}
	-- Walking back to a map we already have: unwind rather than grow.
	for i = #mapChain, 1, -1 do
		if sameMap(mapChain[i], cur) then
			for j = #mapChain, i + 1, -1 do
				mapChain[j] = nil
			end
			return
		end
	end
	mapChain[#mapChain + 1] = cur
end

function Nav.resetMapChain()
	mapChain = {}
	noteCurrentMap()
end

-- =================================================================== steps

local function beginStep(dir)
	local x, y = Nav.playerPos()
	S.stepDir, S.stepX, S.stepY, S.stepFrames = dir, x, y, 0
	io_.hold(dirKey(dir))
end

-- Returns "moving" | "done" | "blocked".
--
-- One call to this moves the player exactly ONE tile. That needs the key
-- released the instant the coordinates change: the game reads the held
-- direction again as soon as it finishes centring on a tile, so a key still
-- down at that moment starts another step immediately and the player runs on
-- for two or three tiles. Releasing early is safe -- the game always finishes
-- the tile it has started.
local function updateStep()
	S.stepFrames = S.stepFrames + 1
	local x, y = Nav.playerPos()
	if not x then
		return "blocked"
	end
	if x ~= S.stepX or y ~= S.stepY then
		io_.release()
		if playerSettled() then
			return "done"
		end
		return "moving"
	end
	if S.stepFrames > STEP_TURN_FRAMES then
		return "blocked"
	end
	if S.stepFrames > STEP_MAX_FRAMES then
		return "blocked"
	end
	io_.hold(dirKey(S.stepDir))
	return "moving"
end

-- ==================================================================== task

local function fail(why)
	io_.release()
	S.phase, S.fail = "failed", why
end

local function finish()
	io_.release()
	S.phase = "done"
end

local function planFor(goal)
	local g = readGrid()
	if not g then
		return nil, "no_map"
	end
	local x, y = Nav.playerPos()
	if not x then
		return nil, "no_player"
	end
	refreshWarpTiles()
	return g, x, y
end

local function mapKey()
	local g, n = currentMap()
	return g and (g .. "." .. n) or nil
end

local OPP_DIR = { [1] = 2, [2] = 1, [3] = 4, [4] = 3 }

-- Tiles the player can stand on to trigger `warp`, added to `out` as
-- "x,y" -> warp.
--
-- Door tiles (Pokemon Center, houses) have their collision bits set: you can
-- never step ONTO one. The game fires the warp when you stand on the tile in
-- front and press toward the door. So if the warp tile is enterable, stepping
-- onto it is the goal; otherwise the goal is each walkable neighbour, and
-- the push phase presses toward the door from there.
local function warpApproaches(g, warp, out)
	local wx, wy = warp.x, warp.y
	for i = 1, 4 do
		local D = DIRS[i]
		if canStep(g, wx - D.dx, wy - D.dy, wx, wy, D.d) then
			out[wx .. "," .. wy] = warp
			return
		end
	end
	for i = 1, 4 do
		local D = DIRS[i]
		local fx, fy = wx - D.dx, wy - D.dy
		-- A walkable tile whose neighbour in direction D is the door.
		if canStep(g, wx, wy, fx, fy, OPP_DIR[D.d]) then
			out[fx .. "," .. fy] = warp
		end
	end
end

-- Plan a walk to the nearest of `warps`, then a push into it. Returns false
-- if none can be approached.
local function planWalkToWarps(g, x, y, warps)
	local goals = {}
	for _, w in ipairs(warps) do
		warpApproaches(g, w, goals)
	end
	if next(goals) == nil then
		return false
	end
	local path, gx, gy = bfsPreferAvoiding(g, x, y, function(_, px, py)
		return goals[px .. "," .. py] ~= nil
	end, isEncounter)
	if not path then
		return false
	end
	local w = goals[gx .. "," .. gy]
	S.warpX, S.warpY, S.pushMap = w.x, w.y, mapKey()
	S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "push"
	return true
end

-- ======================================================= map graph / routing
-- Learned passively from every map the player is ever on (see Nav.observe),
-- so it covers maps walked by hand as well as by this script.
local mapGraph, observedKey = {}, nil

local function connectionsOfCurrentMap()
	local out = {}
	local connections = mem.read32(OW.MAP_HEADER + OW.MH_CONNECTIONS)
	if connections == 0 then
		return out
	end
	local count = mem.read32(connections)
	local arr = mem.read32(connections + 4)
	if arr == 0 or count <= 0 or count > 16 then
		return out
	end
	for i = 0, count - 1 do
		local base = arr + i * 12
		out[#out + 1] = { g = mem.read8(base + 8), n = mem.read8(base + 9) }
	end
	return out
end

-- Call every frame; does real work only when the map changes.
function Nav.observe()
	-- Mid warp/door transition the saveblock location and the map header can
	-- disagree for a few frames; wait until the field is live.
	if fieldLocked() then
		return
	end
	local g, n = currentMap()
	if not g then
		return
	end
	local key = g .. "." .. n
	if key == observedKey then
		return
	end
	observedKey = key
	local out, seen = {}, {}
	local function add(tg, tn)
		local k = tg .. "." .. tn
		if not seen[k] and k ~= key then
			seen[k] = true
			out[#out + 1] = { g = tg, n = tn }
		end
	end
	for _, w in ipairs(Nav.mapWarps()) do
		add(w.group, w.num)
	end
	for _, c in ipairs(connectionsOfCurrentMap()) do
		add(c.g, c.n)
	end
	mapGraph[key] = out
end

-- A wild battle just started from the overworld: this map is an encounter
-- map, whoever walked here.
function Nav.noteBattleMap()
	local g, n = currentMap()
	if g then
		huntMap = { g = g, n = n }
	end
end

-- First map to head for on the shortest known route from `from` to `to`.
local function routeToMap(from, to)
	local fk, tk = from.g .. "." .. from.n, to.g .. "." .. to.n
	if fk == tk then
		return nil
	end
	local first, queue, qh = { [fk] = false }, { fk }, 1
	while qh <= #queue do
		local cur = queue[qh]
		qh = qh + 1
		for _, nb in ipairs(mapGraph[cur] or {}) do
			local k = nb.g .. "." .. nb.n
			if first[k] == nil then
				first[k] = (cur == fk) and nb or first[cur]
				if k == tk then
					return first[k]
				end
				queue[#queue + 1] = k
			end
		end
	end
	return nil
end

-- One hop back along the maps we walked through (warp first, then a map
-- connection). Returns true if a walk was planned.
local function planStepBack(g, x, y)
	local prev = mapChain[#mapChain - 1]
	if not prev then
		return false
	end
	local pw = warpTo(prev.g, prev.n)
	if pw and planWalkToWarps(g, x, y, { pw }) then
		return true
	end
	local edge = Nav.edgeTowardMap(g, prev.g, prev.n)
	if edge then
		local epath = bfsPreferAvoiding(g, x, y, edge, isEncounter)
		if epath then
			S.path, S.pathIdx, S.phase, S.after = interleavePath(epath, g, x, y), 1, "walk", "replan"
			return true
		end
	end
	return false
end

-- Head for the last map where wild battles happened (cave, route...), one
-- map hop at a time over the learned map graph. Falls back to retracing the
-- chain of maps we walked out through.
local function planReturnToHunt(g, x, y)
	if not huntMap then
		return false
	end
	local cg, cn = currentMap()
	if not cg or (cg == huntMap.g and cn == huntMap.n) then
		return false
	end
	local hop = routeToMap({ g = cg, n = cn }, huntMap)
	if hop then
		local w = warpTo(hop.g, hop.n)
		if w and planWalkToWarps(g, x, y, { w }) then
			return true
		end
		local edge = Nav.edgeTowardMap(g, hop.g, hop.n)
		if edge then
			local path = bfsPreferAvoiding(g, x, y, edge, isEncounter)
			if path then
				S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "replan"
				return true
			end
		end
	end
	for i = 1, #mapChain - 1 do
		if sameMap(mapChain[i], huntMap) then
			return planStepBack(g, x, y)
		end
	end
	return false
end

-- Choose the next target for the HUNT goal (grass, cave floor, any tile whose
-- behavior is in OW.ENCOUNTER_BEHAVIORS).
local function planHunt()
	local g, x, y = planFor()
	if not g then
		return fail(x)
	end

	S.triedMaps[mapKey() or "?"] = true
	if planReturnToHunt(g, x, y) then
		return
	end

	if isEncounter(g, x, y) then
		-- wanderTotal is deliberately NOT reset here: coming back to grass
		-- after drifting off must not restart the no-encounter timeout, or a
		-- patch that never produces one would loop forever.
		S.phase = "wander"
		return
	end

	local path = bfs(g, x, y, isEncounter)
	if path then
		S.path, S.pathIdx, S.phase = interleavePath(path, g, x, y), 1, "walk"
		S.after = "wander"
		return
	end

	-- No encounter tiles on this map. First go back the way we came (out of a
	-- Center and on toward the route); only then try the nearest warp. Taking
	-- the nearest warp first re-enters the Center we just left, forever.
	if planStepBack(g, x, y) then
		return
	end
	-- Last resort: any warp we have not already been through this task. Never
	-- re-enter a map we tried (the Pokemon Center loop).
	local fresh = {}
	for _, w in ipairs(Nav.mapWarps()) do
		if not S.triedMaps[w.group .. "." .. w.num] then
			fresh[#fresh + 1] = w
		end
	end
	if #fresh > 0 and planWalkToWarps(g, x, y, fresh) then
		return
	end
	fail(huntMap and "no_encounter_tiles" or "no_hunt_spot")
end

-- Choose the next target for the HEAL goal.
--
-- Three situations, checked in order:
--   1. A nurse is on this map -> we are inside a Pokemon Center; go talk.
--   2. We are on the heal map -> walk to the Center's door and go in.
--   3. Otherwise -> work back toward the heal map.
local function planHeal()
	local hg, hn, hx, hy = healLocation()
	if not hg then
		return fail("no_heal_location")
	end
	local cg, cn = currentMap()
	if not cg then
		return fail("no_map")
	end

	-- Right after a warp the nurse's object may not be loaded yet. Without
	-- this wait the planner decides "not a Center" and walks straight out.
	if not nurseTile() and not (cg == hg and cn == hn) then
		local key = mapKey()
		if S.nurseWaitKey ~= key then
			S.nurseWaitKey, S.nurseWaitLeft = key, 30
		end
		if S.nurseWaitLeft > 0 then
			S.nurseWaitLeft = S.nurseWaitLeft - 1
			return
		end
	end

	local g, x, y = planFor()
	if not g then
		return fail(x)
	end

	-- 1. Inside a Pokemon Center. Walk to a tile the nurse can be talked to
	-- from: she stands behind a counter, so that is up to NURSE_TALK_RANGE
	-- tiles away in a straight line, not an adjacent tile.
	local nx, ny = nurseTile()
	if nx then
		local R = OW.NURSE_TALK_RANGE
		local function canTalkFrom(_, px, py)
			if px == nx and math.abs(py - ny) >= 1 and math.abs(py - ny) <= R then
				return true
			end
			if py == ny and math.abs(px - nx) >= 1 and math.abs(px - nx) <= R then
				return true
			end
			return false
		end
		if canTalkFrom(g, x, y) then
			S.nurseX, S.nurseY = nx, ny
			S.phase, S.talkFrames = "heal_face", 0
			return
		end
		local path = bfs(g, x, y, canTalkFrom)
		if path then
			S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "replan"
			return
		end
		return fail("counter_unreachable")
	end

	-- 2. On the heal map: the Center's door is the warp beside the respawn tile.
	if cg == hg and cn == hn then
		local door = pokemonCenterDoor(hx + MAP_OFFSET, hy + MAP_OFFSET)
		if not door then
			return fail("no_center_door")
		end
		if planWalkToWarps(g, x, y, { door }) then
			return
		end
		return fail("center_door_unreachable")
	end

	-- 3. Some other map. A direct warp to the heal map, if one exists.
	local warp = warpTo(hg, hn)
	if warp and planWalkToWarps(g, x, y, { warp }) then
		return
	end

	-- A map connection straight to the heal map (a route bordering the city).
	local healEdge = Nav.edgeTowardMap(g, hg, hn)
	if healEdge then
		local path = bfsPreferAvoiding(g, x, y, healEdge, isEncounter)
		if path then
			S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "replan"
			return
		end
		-- The connection is there but no walkable route reaches that edge.
		-- Distinguishing this from "no connection at all" matters: they have
		-- completely different causes.
		return fail("center_edge_unreachable")
	end

	-- Otherwise step back along the chain of maps we walked out through.
	local prev = mapChain[#mapChain - 1]
	if prev then
		local pwarp = warpTo(prev.g, prev.n)
		if pwarp and planWalkToWarps(g, x, y, { pwarp }) then
			return
		end
		local edge = Nav.edgeTowardMap(g, prev.g, prev.n)
		if edge then
			local path = bfsPreferAvoiding(g, x, y, edge, isEncounter)
			if path then
				S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "replan"
				return
			end
		end
	end
	fail("no_route_to_center")
end

-- Goal predicate matching any tile past the current map's edge in the
-- direction of a connected map. Stepping there crosses the connection.
function Nav.edgeTowardMap(g, group, num)
	local connections = mem.read32(OW.MAP_HEADER + OW.MH_CONNECTIONS)
	if connections == 0 then
		return nil
	end
	local count = mem.read32(connections)
	local arr = mem.read32(connections + 4)
	if arr == 0 or count <= 0 or count > 16 then
		return nil
	end

	local layout = mem.read32(OW.MAP_HEADER + OW.MH_MAP_LAYOUT)
	local mw = mem.read32(layout)
	local mh = mem.read32(layout + 4)

	for i = 0, count - 1 do
		local base = arr + i * 12
		local dir = mem.read8(base)
		local cg = mem.read8(base + 8)
		local cn = mem.read8(base + 9)
		if cg == group and cn == num then
			if dir == 2 then -- north
				return function(_, _, py)
					return py < MAP_OFFSET
				end
			elseif dir == 1 then -- south
				return function(_, _, py)
					return py >= MAP_OFFSET + mh
				end
			elseif dir == 3 then -- west
				return function(_, px, _)
					return px < MAP_OFFSET
				end
			elseif dir == 4 then -- east
				return function(_, px, _)
					return px >= MAP_OFFSET + mw
				end
			end
		end
	end
	return nil
end

-- A step the game refused. Logs everything canStep() looked at, so the rule
-- it is missing can be identified, and remembers the edge after two failures
-- (one could be an NPC wandering through or a text box).
local function noteBlocked(dir, fx, fy)
	if not dir or not fx then
		return
	end
	local D
	for i = 1, 4 do
		if DIRS[i].d == dir then
			D = DIRS[i]
		end
	end
	if not D then
		return
	end
	local key = fx .. "," .. fy .. ">" .. dir
	blockedCounts[key] = (blockedCounts[key] or 0) + 1
	if blockedCounts[key] >= 2 then
		blockedEdges[key] = true
	end
	local g = readGrid()
	if not g then
		return
	end
	local tx, ty = fx + D.dx, fy + D.dy
	local to = cellAt(g, tx, ty) or 0
	local function b(x, y)
		local v = behaviorAt(g, x, y)
		return v and string.format("0x%02X", v) or "nil"
	end
	log(string.format(
		"blocked step %s (%d,%d)->(%d,%d) dir=%d #%d: cell=0x%04X metatile=%d collision=%d tileElev=%d playerElev=%d behavior from=%s to=%s npc=%s",
		tostring(mapKey()), fx, fy, tx, ty, dir, blockedCounts[key],
		to, to & OW.MAPGRID_METATILE_ID_MASK,
		(to & OW.MAPGRID_COLLISION_MASK) >> OW.MAPGRID_COLLISION_SHIFT,
		(to & OW.MAPGRID_ELEVATION_MASK) >> OW.MAPGRID_ELEVATION_SHIFT,
		g.elevation or -1, b(fx, fy), b(tx, ty), tostring(g.objects[tx + ty * g.w] == true)
	))
end

local function replan()
	S.replans = (S.replans or 0) + 1
	if S.replans > MAX_REPLANS then
		return fail("replan_limit")
	end
	S.path, S.pathIdx = nil, nil
	S.phase = "plan"
end

local OPPOSITE_DIR = { [1] = 2, [2] = 1, [3] = 4, [4] = 3 }

-- Step around inside a patch of grass until something jumps out.
--
-- Only stepping ONTO an encounter tile rolls for a wild Pokemon, so the
-- measure that matters is how many steps LAND on grass, not how close the
-- player is to some. The loop is anchored rather than greedy:
--
--   * Every time the player stands on grass, that tile is remembered as the
--     anchor. It is the patch we are working, not just "some grass".
--   * On grass, step to a neighbouring grass tile, rotating which direction
--     is tried first so the walk spreads over the patch instead of pacing
--     the same two tiles. Reversing is allowed -- a two-tile patch has no
--     third tile, and bouncing between two rolls on every step.
--   * Off grass, walk back to the anchor. Checking only the four adjacent
--     tiles was not enough: one step too far and nothing pulled the player
--     back, so it drifted. Re-planning for the "nearest" grass instead of
--     the anchor is also wrong, because that can pick a different patch
--     each time and ping-pong between them.
--   * The anchor is only abandoned when it is further away than
--     WANDER_ANCHOR_RANGE, which means something unexpected moved the
--     player; then, and only then, look for a fresh patch.
local function updateWander()
	local hg, hn = currentMap()
	if hg then
		huntMap = { g = hg, n = hn }
	end
	S.wanderTotal = (S.wanderTotal or 0) + 1
	if S.wanderTotal > WANDER_MAX_FRAMES then
		return fail("no_encounter")
	end

	if S.stepDir then
		local r = updateStep()
		if r == "moving" then
			return
		end
		io_.release()
		local bd, bx, by = S.stepDir, S.stepX, S.stepY
		S.stepDir = nil
		if r == "blocked" then
			noteBlocked(bd, bx, by)
			S.lastWanderBack = nil
			S.wanderSpin = (S.wanderSpin or 0) + 1 -- try a different way next
		else
			S.wanderSteps = (S.wanderSteps or 0) + 1
		end
		return
	end

	local g = readGrid()
	if not g then
		return fail("no_map")
	end
	local x, y = Nav.playerPos()
	if not x then
		return fail("no_player")
	end

	if isEncounter(g, x, y) then
		S.anchorX, S.anchorY = x, y
		S.wanderOnGrass = (S.wanderOnGrass or 0) + 1

		-- Rotate the starting direction so the walk covers the patch.
		S.wanderSpin = ((S.wanderSpin or 0) + 1) % 4
		local grassDir, reverseDir, anyDir
		for k = 0, 3 do
			local D = DIRS[((S.wanderSpin + k) % 4) + 1]
			local nx, ny = x + D.dx, y + D.dy
			if canStep(g, x, y, nx, ny, D.d) then
				anyDir = anyDir or D.d
				if isEncounter(g, nx, ny) then
					if D.d == S.lastWanderBack then
						reverseDir = reverseDir or D.d
					else
						grassDir = grassDir or D.d
					end
				end
			end
		end
		-- A lone grass tile with no grass neighbour: step off, and the anchor
		-- brings us straight back, so the next step lands on it again.
		local dir = grassDir or reverseDir or anyDir
		if not dir then
			return fail("wander_stuck")
		end
		S.lastWanderBack = OPPOSITE_DIR[dir]
		return beginStep(dir)
	end

	-- Off the grass. Head back to the tile we were last standing on.
	S.lastWanderBack = nil
	if not S.anchorX then
		S.phase = "plan"
		return
	end
	local dist = math.abs(x - S.anchorX) + math.abs(y - S.anchorY)
	if dist > WANDER_ANCHOR_RANGE then
		-- Something moved us a long way; this patch is no longer the one.
		S.anchorX, S.anchorY = nil, nil
		S.phase = "plan"
		return
	end

	local ax, ay = S.anchorX, S.anchorY
	local path = bfs(g, x, y, function(_, px, py)
		return px == ax and py == ay
	end)
	if not path then
		-- The anchor became unreachable (an NPC parked on it, say).
		S.anchorX, S.anchorY = nil, nil
		S.phase = "plan"
		return
	end
	if #path == 0 then
		-- Already standing on the anchor. bfs returns an empty path for
		-- "you are already there", which is not the same as "no route" --
		-- treating the two alike threw away a perfectly good anchor and
		-- sent the script back to a full search for the nearest patch.
		S.phase = "wander"
		return
	end
	S.path, S.pathIdx, S.phase, S.after = interleavePath(path, g, x, y), 1, "walk", "wander"
end

-- Face the counter and talk until the party comes back at full HP. Tapping A
-- and checking real HP is self-verifying: no fixed number of presses, and no
-- assumption about how many text boxes the nurse shows.
local function updateHealTalk()
	S.talkFrames = S.talkFrames + 1
	if partyFullyHealed() then
		io_.release()
		S.phase = "healed"
		return
	end
	if S.talkFrames > 60 * 40 then
		return fail("heal_timeout")
	end
	local t = S.talkFrames % TALK_TAP_PERIOD
	if t < 2 then
		io_.hold(KEY.A)
	else
		io_.release()
	end
end

-- Press toward the warp until the map changes. Covers doors (stand in front
-- and press at them) and exit mats (standing on the warp tile does nothing
-- until you press toward the edge). If the map already changed on arrival
-- (stairs, cave mouths) this finishes at once.
local PUSH_FRAMES = 60 * 3
local PUSH_MAX_TRIES = 3

local function updatePush()
	if mapKey() ~= S.pushMap then
		io_.release()
		S.pushDir, S.pushTries = nil, 0
		S.phase = "plan"
		return
	end

	S.pushFrames = (S.pushFrames or 0) + 1
	if S.pushFrames > PUSH_FRAMES then
		io_.release()
		S.pushDir, S.pushFrames = nil, 0
		S.pushTries = (S.pushTries or 0) + 1
		if S.pushTries >= PUSH_MAX_TRIES then
			return fail("warp_no_effect")
		end
		return replan()
	end

	if not S.pushDir then
		local px, py = Nav.playerPos()
		if not px then
			return fail("no_player")
		end
		if px == S.warpX and py == S.warpY then
			-- On the warp tile itself: press toward the side with nothing
			-- walkable behind it (south first -- mats face the exit).
			local g = readGrid()
			S.pushDir = 1
			if g then
				local byDir = {}
				for i = 1, 4 do
					byDir[DIRS[i].d] = DIRS[i]
				end
				for _, d in ipairs({ 1, 3, 4, 2 }) do
					local D = byDir[d]
					if not canStep(g, px, py, px + D.dx, py + D.dy, d) then
						S.pushDir = d
						break
					end
				end
			end
		else
			for i = 1, 4 do
				local D = DIRS[i]
				if px + D.dx == S.warpX and py + D.dy == S.warpY then
					S.pushDir = D.d
				end
			end
			if not S.pushDir then
				-- Not next to the warp (displaced by an NPC, say): replan.
				S.pushFrames = 0
				return replan()
			end
		end
	end
	io_.hold(dirKey(S.pushDir))
end

-- =================================================================== debug

-- ASCII picture of the walkable grid, for checking the addresses and the
-- passability rules by eye before trusting anything built on top of them.
--   @ player   . walkable   # wall   " grass   + warp   ^ ledge   o npc
function Nav.dump()
	local g = readGrid()
	if not g then
		return "MAP:ERROR no map data\n"
	end
	local px, py = Nav.playerPos()
	local cg, cn = currentMap()
	local hg, hn, hx, hy = healLocation()
	local out = {
		string.format(
			"MAP %dx%d map=%d.%d player=%d,%d elev=%d heal=%d.%d@%d,%d",
			g.w,
			g.h,
			cg or -1,
			cn or -1,
			px or -1,
			py or -1,
			g.elevation,
			hg or -1,
			hn or -1,
			(hx or -1) + MAP_OFFSET,
			(hy or -1) + MAP_OFFSET
		),
	}
	for y = 0, g.h - 1 do
		local row = {}
		for x = 0, g.w - 1 do
			local ch
			local cell = cellAt(g, x, y)
			local b = behaviorAt(g, x, y)
			if px == x and py == y then
				ch = "@"
			elseif g.objects[x + y * g.w] then
				ch = "o"
			elseif cell == nil or (cell & OW.MAPGRID_COLLISION_MASK) ~= 0 then
				ch = "#"
			elseif OW.ENCOUNTER_BEHAVIORS[b] then
				ch = '"'
			elseif OW.WARP_BEHAVIORS[b] then
				ch = "+"
			elseif OW.LEDGE_BEHAVIORS[b] then
				ch = "^"
			elseif OW.BLOCKED_NORTH[b] or OW.BLOCKED_SOUTH[b] or OW.BLOCKED_EAST[b] or OW.BLOCKED_WEST[b] then
				ch = ":"
			else
				ch = "."
			end
			row[#row + 1] = ch
		end
		out[#out + 1] = table.concat(row)
	end
	local warps = {}
	for _, wp in ipairs(Nav.mapWarps()) do
		warps[#warps + 1] = string.format("(%d,%d)->%d.%d", wp.x, wp.y, wp.group, wp.num)
	end
	out[#out + 1] = "warps: " .. table.concat(warps, " ")

	-- Connections to neighbouring maps, and the grid row/column that counts
	-- as "over the edge" into each. Walking to one of those tiles is how the
	-- script crosses from a route into the town with the Pokemon Center.
	local layout = mem.read32(OW.MAP_HEADER + OW.MH_MAP_LAYOUT)
	local mw = (layout ~= 0) and mem.read32(layout) or -1
	local mh = (layout ~= 0) and mem.read32(layout + 4) or -1
	out[#out + 1] = string.format(
		"map size: %dx%d (edges: north y<%d, south y>=%d, west x<%d, east x>=%d)",
		mw,
		mh,
		MAP_OFFSET,
		MAP_OFFSET + mh,
		MAP_OFFSET,
		MAP_OFFSET + mw
	)

	local connections = mem.read32(OW.MAP_HEADER + OW.MH_CONNECTIONS)
	if connections == 0 then
		out[#out + 1] = "connections: <none> (pointer is null)"
	else
		local count = mem.read32(connections)
		local arr = mem.read32(connections + 4)
		local names = { [1] = "south", [2] = "north", [3] = "west", [4] = "east" }
		local parts = { string.format("count=%d arr=0x%08X", count, arr) }
		if arr ~= 0 and count > 0 and count <= 16 then
			for i = 0, count - 1 do
				local base = arr + i * 12
				local dir = mem.read8(base)
				parts[#parts + 1] =
					string.format("%s(%d)->%d.%d", names[dir] or "?", dir, mem.read8(base + 8), mem.read8(base + 9))
			end
		end
		out[#out + 1] = "connections: " .. table.concat(parts, " ")
	end

	-- Elevation census. A step is only legal onto a tile whose elevation
	-- matches the player's (0 and 15 always pass), so a route that looks
	-- walkable in the picture above can still be refused. If the walkable
	-- tiles are mostly at an elevation the player isn't on, that's why.
	local elevs = {}
	for ey = 0, g.h - 1 do
		for ex = 0, g.w - 1 do
			local c = cellAt(g, ex, ey)
			if c and (c & OW.MAPGRID_COLLISION_MASK) == 0 then
				local e = (c & OW.MAPGRID_ELEVATION_MASK) >> OW.MAPGRID_ELEVATION_SHIFT
				elevs[e] = (elevs[e] or 0) + 1
			end
		end
	end
	local eparts = {}
	for e = 0, 15 do
		if elevs[e] then
			eparts[#eparts + 1] = string.format("%d:%d", e, elevs[e])
		end
	end
	out[#out + 1] = string.format("player elev %d; walkable tiles by elev: %s", g.elevation, table.concat(eparts, " "))
	return table.concat(out, "\n") .. "\n"
end

-- Why a plan did or didn't find anything, reported from the live game. The
-- ASCII map can't show elevation, object events or directional walls, and
-- those are exactly the things that make a tile that looks walkable refuse a
-- step, so this reports each one separately instead of just "no_grass".
function Nav.debug()
	local out = {}
	local function add(s)
		out[#out + 1] = s
	end

	local g = readGrid()
	if not g then
		add("grid: UNREADABLE (gBackupMapLayout or the tilesets came back empty)")
		return table.concat(out, "\n") .. "\n"
	end
	local x, y = Nav.playerPos()
	if not x then
		add("player: UNREADABLE")
		return table.concat(out, "\n") .. "\n"
	end
	refreshWarpTiles()

	local nObj = 0
	for _ in pairs(g.objects) do
		nObj = nObj + 1
	end
	add(
		string.format(
			"grid %dx%d  player (%d,%d) elev %d  settled %s  objects %d  field_locked %s",
			g.w,
			g.h,
			x,
			y,
			g.elevation,
			tostring(playerSettled()),
			nObj,
			tostring(fieldLocked())
		)
	)

	-- What the planner can see at all.
	local enc, warps = 0, 0
	for ty = 0, g.h - 1 do
		for tx = 0, g.w - 1 do
			if isEncounter(g, tx, ty) then
				enc = enc + 1
			end
			if isWarpTile(g, tx, ty) then
				warps = warps + 1
			end
		end
	end
	add(string.format("tiles the planner sees: encounter %d, warp %d", enc, warps))
	add(
		string.format(
			"standing on: behavior 0x%02X  encounter %s",
			behaviorAt(g, x, y) or -1,
			tostring(isEncounter(g, x, y))
		)
	)

	-- Each first step, with the reason it is refused.
	local names = { [1] = "south", [2] = "north", [3] = "west", [4] = "east" }
	local parts = {}
	for i = 1, 4 do
		local D = DIRS[i]
		local tx, ty = x + D.dx, y + D.dy
		local why = "ok"
		local c = cellAt(g, tx, ty)
		if c == nil then
			why = "off-grid"
		elseif (c & OW.MAPGRID_METATILE_ID_MASK) == OW.MAPGRID_UNDEFINED then
			why = "undefined"
		elseif (c & OW.MAPGRID_COLLISION_MASK) ~= 0 then
			why = "collision"
		elseif g.objects[tx + ty * g.w] then
			why = "object"
		elseif not canStep(g, x, y, tx, ty, D.d) then
			local e = (c & OW.MAPGRID_ELEVATION_MASK) >> OW.MAPGRID_ELEVATION_SHIFT
			why = string.format("blocked(elev %d vs %d, beh 0x%02X)", e, g.elevation, behaviorAt(g, tx, ty) or -1)
		end
		parts[#parts + 1] = string.format("%s=%s", names[D.d], why)
	end
	add("first step: " .. table.concat(parts, " "))

	-- How far the planner can actually get, and to what.
	local reach, queue, qh, qt = { [x + y * g.w] = true }, { x + y * g.w }, 1, 1
	local n = 1
	while qh <= qt do
		local cur = queue[qh]
		qh = qh + 1
		local cx, cy = cur % g.w, cur // g.w
		for i = 1, 4 do
			local D = DIRS[i]
			local nx, ny = cx + D.dx, cy + D.dy
			local nidx = nx + ny * g.w
			if
				nx >= 0
				and ny >= 0
				and nx < g.w
				and ny < g.h
				and not reach[nidx]
				and canStep(g, cx, cy, nx, ny, D.d)
			then
				reach[nidx] = true
				n = n + 1
				qt = qt + 1
				queue[qt] = nidx
			end
		end
	end
	add(string.format("reachable tiles: %d", n))

	local path, gx, gy = bfs(g, x, y, isEncounter)
	if path then
		add(string.format("nearest encounter tile: (%d,%d), %d steps", gx, gy, #path))
	else
		-- Say how close the nearest one is in a straight line, so we can tell
		-- "there is none" from "there is one but nothing can reach it".
		local bd, bx, by
		for ty = 0, g.h - 1 do
			for tx = 0, g.w - 1 do
				if isEncounter(g, tx, ty) then
					local d = math.abs(tx - x) + math.abs(ty - y)
					if not bd or d < bd then
						bd, bx, by = d, tx, ty
					end
				end
			end
		end
		if bd then
			add(string.format("NO ROUTE to any encounter tile; nearest is (%d,%d), %d tiles away", bx, by, bd))
		else
			add("there are NO encounter tiles anywhere on this map")
		end
	end

	local wpath, wx, wy = bfs(g, x, y, isWarpTile)
	add(wpath and string.format("nearest warp: (%d,%d), %d steps", wx, wy, #wpath) or "no reachable warp")
	return table.concat(out, "\n") .. "\n"
end

-- ================================================================== public

-- goal: "HUNT"  (find encounter tiles -- grass or cave floor -- and keep
--                stepping until a battle starts; "GRASS" is an alias)
--       "HEAL"  (walk to the Pokemon Center, heal, then return to the hunt map)
function Nav.start(goal)
	if goal == "GRASS" then
		goal = "HUNT"
	end
	noteCurrentMap()
	blockedEdges, blockedCounts = {}, {}
	S = { goal = goal, phase = "plan", frames = 0, replans = 0, triedMaps = {} }
	return true
end

function Nav.cancel()
	if S then
		io_.release()
	end
	S = nil
end

function Nav.active()
	return S ~= nil and S.phase ~= "done" and S.phase ~= "failed"
end

function Nav.status()
	if not S then
		return "IDLE"
	end
	if S.phase == "done" then
		return "ARRIVED"
	end
	if S.phase == "failed" then
		return "FAILED:" .. (S.fail or "unknown")
	end
	if S.phase == "battle" then
		return "BATTLE"
	end
	-- Wandering reports its counters, because "is this working?" is really
	-- "are steps landing on grass?" and that is otherwise invisible.
	if S.phase == "wander" or (S.after == "wander" and S.phase == "walk") then
		return string.format(
			"WALKING:%s steps=%d on_grass=%d anchor=%s",
			S.phase,
			S.wanderSteps or 0,
			S.wanderOnGrass or 0,
			S.anchorX and (S.anchorX .. "," .. S.anchorY) or "none"
		)
	end
	local nb = 0
	for _ in pairs(blockedEdges) do
		nb = nb + 1
	end
	return "WALKING:" .. S.phase
		.. (huntMap and (" hunt=" .. huntMap.g .. "." .. huntMap.n) or "")
		.. (nb > 0 and (" blocked_edges=" .. nb) or "")
end

-- Called every frame by the bridge while a nav task is running.
-- `inBattle` is the bridge's gMain.inBattle reading; a wild encounter can
-- start in the middle of any step, and when it does the task is over.
-- Phases that already expect the field to be locked and drive their own A
-- presses (the heal conversation). A Match Call interrupting one of THESE
-- would be unusual, but excluding them means the lock check never fights
-- with logic that's deliberately holding a text box open itself.
local SELF_LOCKING_PHASES = { heal_face = true, heal_talk = true }
local LOCK_DEBOUNCE_FRAMES = 20 -- a text box outlasts a door animation

function Nav.update(inBattle)
	if not S or S.phase == "done" or S.phase == "failed" then
		return
	end

	if inBattle then
		io_.release()
		S.phase = "battle"
		return
	end

	-- A Match Call (or any other text box) has taken input away. Walking
	-- steps fail the instant this happens, which otherwise reads as a wall
	-- the planner bumped into -- it would burn replans and eventually give
	-- up, when really it just needs to wait and clear the box.
	-- Debounced: door/warp transitions and one-frame script locks also set the
	-- flag, and none of them are a text box. Only a lock that persists counts.
	if S.phase ~= "dialog" and not SELF_LOCKING_PHASES[S.phase] and fieldLocked() then
		S.lockFrames = (S.lockFrames or 0) + 1
	else
		S.lockFrames = 0
	end
	if S.lockFrames >= LOCK_DEBOUNCE_FRAMES then
		S.lockFrames = 0
		io_.release()
		-- Drop any in-flight step: its start position is stale once the call ends.
		S.stepDir = nil
		S.dialogResume, S.dialogFrames = S.phase, 0
		S.dialogCount = (S.dialogCount or 0) + 1
		local px, py = Nav.playerPos()
		log(string.format(
			"input locked (raw=0x%02X) map=%s pos=%s,%s phase='%s' -- tapping through (#%d)",
			mem.read8(OW.LOCK_FIELD_CONTROLS), tostring(mapKey()), tostring(px), tostring(py), S.phase, S.dialogCount
		))
		S.phase = "dialog"
		return
	end

	if S.phase == "dialog" then
		S.dialogFrames = S.dialogFrames + 1
		if not fieldLocked() then
			-- The box is gone, but the player may have moved, turned, or be
			-- standing somewhere a half-finished step no longer makes sense
			-- for. Replanning from here is simpler and safer than trying to
			-- resume an in-flight step or path.
			io_.release()
			log(string.format("input unlocked after %d frames -- replanning", S.dialogFrames))
			S.stepDir, S.lastWanderBack = nil, nil
			S.path, S.pathIdx, S.after = nil, nil, nil
			S.phase = (S.dialogResume == "heal_face" or S.dialogResume == "heal_talk" or S.dialogResume == "healed")
					and S.dialogResume
				or "plan"
			S.dialogResume = nil
			return
		end
		if S.dialogFrames > 60 * 15 then
			log("input still locked after 15s -- giving up")
			return fail("dialog_stuck")
		end
		-- Tap A first (advances most text), then B (backs out of a menu a
		-- stray A might have opened). Same cadence as the heal conversation.
		local t = S.dialogFrames % TALK_TAP_PERIOD
		if t < 2 then
			io_.hold(KEY.A)
		elseif t < 4 then
			io_.release()
		elseif t < 6 then
			io_.hold(KEY.B)
		else
			io_.release()
		end
		return
	end

	S.frames = S.frames + 1
	if S.frames > TASK_MAX_FRAMES then
		return fail("task_timeout")
	end

	-- Never act part-way through a step the game is still animating.
	if S.phase ~= "walk" and S.phase ~= "wander" and not playerSettled() then
		io_.release()
		return
	end

	if S.phase == "plan" then
		noteCurrentMap()
		if S.goal == "HEAL" and not S.healed then
			planHeal()
		else
			planHunt()
		end
	elseif S.phase == "walk" then
		if S.stepDir then
			local r = updateStep()
			if r == "moving" then
				return
			end
			io_.release()
			local bd, bx, by = S.stepDir, S.stepX, S.stepY
			S.stepDir = nil
			if r == "blocked" then
				noteBlocked(bd, bx, by)
				return replan()
			end
			S.pathIdx = S.pathIdx + 1
		end
		if not S.path or S.pathIdx > #S.path then
			io_.release()
			local after = S.after
			S.path, S.pathIdx, S.after = nil, nil, nil
			if after == "wander" then
				S.phase = "wander"
			elseif after == "push" then
				S.phase, S.pushDir, S.pushFrames = "push", nil, 0
			else
				S.phase = "plan"
			end
			return
		end
		beginStep(S.path[S.pathIdx])
	elseif S.phase == "wander" then
		updateWander(inBattle)
	elseif S.phase == "push" then
		updatePush()
	elseif S.phase == "heal_face" then
		-- A short tap turns the player without moving them off the tile.
		S.talkFrames = S.talkFrames + 1
		if S.talkFrames < 3 then
			local px, py = Nav.playerPos()
			local key = KEY.UP
			if px and S.nurseX then
				if S.nurseY < py then
					key = KEY.UP
				elseif S.nurseY > py then
					key = KEY.DOWN
				elseif S.nurseX < px then
					key = KEY.LEFT
				else
					key = KEY.RIGHT
				end
			end
			io_.hold(key)
		elseif S.talkFrames < 12 then
			io_.release()
		else
			S.phase, S.talkFrames = "heal_talk", 0
		end
	elseif S.phase == "heal_talk" then
		updateHealTalk()
	elseif S.phase == "healed" then
		-- Party is restored; now go find grass again.
		S.healed = true
		S.goal = "HUNT"
		S.phase, S.replans = "plan", 0
	end
end

return Nav
