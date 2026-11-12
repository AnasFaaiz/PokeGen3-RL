-- emerald_config.lua -- per-game config for Pokemon Emerald (USA/Europe,
-- BPEE), consumed by games/battle_bridge.lua. Data only, no logic.
--
-- Every address below comes from pret/pokeemerald's published symbol file
-- (symbols branch, pokeemerald.sym) plus struct layouts in pret's source.
-- The symbol file is trustworthy for your ROM: every address you had
-- already confirmed working by hand (gBattleMons, gPlayerParty,
-- gEnemyParty, the party-menu slot) sits exactly where it says.
--
-- Function addresses (CB2_*, FN_*) are listed as in the .sym file (even).
-- In RAM, Thumb function pointers have bit 0 set (e.g. ...5D instead of
-- ...5C). The bridge compares them with bit 0 ignored.
--
-- To add a new game: copy this file and redo every value for that game.

local M = {}

M.HAS_DOUBLES = true

-- ---------------------------------------------------------------- battle mons
M.BATTLE_MONS_ADDR = 0x02024084 -- gBattleMons[4], 0x58 bytes each
M.BATTLE_MON_STRIDE = 0x58
M.BATTLER_BASE_P1 = 0x02024084 -- battler 0 (player left)
M.BATTLER_BASE_E1 = 0x020240DC -- battler 1 (enemy left)
M.BATTLER_BASE_P2 = 0x02024134 -- battler 2 (player right, doubles)
M.BATTLER_BASE_E2 = 0x0202418C -- battler 3 (enemy right, doubles)

-- struct BattlePokemon offsets (include/pokemon.h).
-- FIXED: level and max_hp were swapped. level is the u8 at 0x2A, maxHP is
-- the u16 at 0x2C. The old max_hp read (u16 at 0x2A) returned
-- level + friendship*256, so the HP-ratio inputs were near zero.
M.BATTLER_OFFSETS = {
	species = 0x00,
	move1 = 0x0C,
	move2 = 0x0E,
	move3 = 0x10,
	move4 = 0x12,
	atk_buff = 0x19,
	def_buff = 0x1A,
	spd_buff = 0x1B, -- statStages[], neutral = 6
	ability = 0x20, -- confirmed real field
	type1 = 0x21,
	type2 = 0x22,
	pp1 = 0x24,
	pp2 = 0x25,
	pp3 = 0x26,
	pp4 = 0x27,
	hp = 0x28,
	level = 0x2A,
	max_hp = 0x2C,
	item = 0x2E,
	personality = 0x48,
	status1 = 0x4C,
	status2 = 0x50,
}

-- ---------------------------------------------------------------- parties
M.PARTY_START = 0x020244EC -- gPlayerParty (PID at +0 of each slot)
M.ENEMY_PARTY_START = 0x02024744 -- gEnemyParty
M.PARTY_STRIDE = 0x64
M.PARTY_HP_OFFSET = 0x56 -- u16 hp inside struct Pokemon
M.PARTY_MAXHP_OFFSET = 0x58 -- u16 maxHP
M.PARTY_HP_BASE = 0x02024542 -- = PARTY_START + 0x56 (kept for compatibility)

-- ---------------------------------------------------------------- battle state
M.BATTLE_TYPE_FLAGS_ADDR = 0x02022FEC -- gBattleTypeFlags
M.BATTLE_TYPE_DOUBLE_BIT = 0x0001 -- BATTLE_TYPE_DOUBLE = 1 << 0
M.BATTLE_OUTCOME_ADDR = 0x0202433A -- gBattleOutcome: 0 = ongoing
M.OUTCOME_WON, M.OUTCOME_LOST, M.OUTCOME_FORFEITED = 1, 2, 9

-- FIXED: the old WEATHER_ADDR (0x02024082) was gCurrentTurnActionNumber,
-- a turn counter. Real weather is the u16 flag set gBattleWeather.
M.WEATHER_ADDR = 0x020243CC
M.WEATHER_RAIN_MASK = 0x0007
M.WEATHER_SAND_MASK = 0x0018
M.WEATHER_SUN_MASK = 0x0060
M.WEATHER_HAIL_MASK = 0x0080

-- gMain (0x030022C0). callback2 at +0x004, inBattle is bit 1 of byte +0x439
-- (include/main.h). inBattle stays set until the game is fully back on the
-- field, including evolution after battle.
M.MAIN_CALLBACK2_ADDR = 0x030022C4
M.GMAIN_INBATTLE_ADDR = 0x030026F9
M.GMAIN_INBATTLE_MASK = 0x02

-- Which handler each battler's controller is running (u32[4]). When the
-- player's entry is HandleInputChooseAction, the game is waiting for input.
M.CONTROLLER_FUNCS_ADDR = 0x03005D60 -- gBattlerControllerFuncs
M.FN_CHOOSE_ACTION = 0x08057588 -- HandleInputChooseAction (player)
M.FN_CHOOSE_MOVE = 0x08057BFC -- HandleInputChooseMove
M.FN_CHOOSE_TARGET = 0x08057824 -- HandleInputChooseTarget (doubles)
M.FN_YESNO_INPUT = 0x080599D4 -- PlayerHandleYesNoInput

-- Menu cursors. Action and move menus are 2x2 grids: bit 0 = column,
-- bit 1 = row (battle_controller_player.c). Action: 0 Fight, 1 Bag,
-- 2 Pokemon, 3 Run.
M.ACTION_CURSOR_ADDR = 0x020244AC -- gActionSelectionCursor[4]
M.MOVE_CURSOR_ADDR = 0x020244B0 -- gMoveSelectionCursor[4]
M.TARGET_CURSOR_ADDR = 0x03005D74 -- gMultiUsePlayerCursor (target battler)

-- Battle script engine: the byte at *gBattlescriptCurrInstr is the opcode
-- currently running. While a prompt waits for input, it stays on that
-- opcode. Opcode numbers from gBattleScriptingCommandsTable.
M.BATTLESCRIPT_PTR_ADDR = 0x02024214
M.OP_LEARNMOVE_YESNO = 0x5A -- "Delete a move to make room?"  -> B (No)
M.OP_STOPLEARNING_YESNO = 0x5B -- "Stop learning the move?"      -> A (Yes)
M.OP_YESNOBOX = 0x67 -- "Use next Pokemon?" / "Change Pokemon?"
M.OP_LVLUP_BOX = 0x6C -- level-up stats box, any key advances

-- ---------------------------------------------------------------- party menu
M.PARTY_MENU_SLOT_ADDR = 0x0203CED1 -- gPartyMenu.slotId (offset 9)
M.PARTY_MENU_ACTION_ADDR = 0x0203CED3 -- gPartyMenu.action (offset 11)
M.PARTY_ACTION_SEND_OUT = 1
M.PARTY_ACTION_SWITCH = 8

-- ---------------------------------------------------------------- callbacks
M.CB2_OVERWORLD = 0x08085E5C
M.CB2_BATTLE_MAIN = 0x08038420 -- BattleMainCB2
M.CB2_PARTY_MENU = 0x081B01B0 -- CB2_UpdatePartyMenu
M.CB2_EVOLUTION_LOAD = 0x0813DD7C -- CB2_EvolutionSceneLoadGraphics
M.CB2_EVOLUTION_UPDATE = 0x0813E3A4 -- CB2_EvolutionSceneUpdate

-- ================================================================= overworld
-- Everything below drives scripted navigation (games/overworld_nav.lua).
-- Addresses from pokeemerald.sym; struct offsets from include/global.fieldmap.h
-- and include/global.h. Symbol sizes match the structs exactly (gObjectEvents
-- is 0x240 = 16 * 0x24, gPlayerAvatar 0x24, gBackupMapLayout 0x0C), which is a
-- good independent check that these are the right symbols.
M.OW = {
	-- struct BackupMapLayout { s32 width; s32 height; u16 *map; }
	BACKUP_MAP_LAYOUT = 0x03005DC0,
	MAP_HEADER = 0x02037318, -- gMapHeader
	OBJECT_EVENTS = 0x02037350, -- gObjectEvents[16]
	PLAYER_AVATAR = 0x02037590, -- gPlayerAvatar
	SAVEBLOCK1_PTR = 0x03005D8C, -- gSaveBlock1Ptr (pointer, not struct)

	-- sLockFieldControls (script.c): nonzero whenever the player's input is
	-- taken away -- a Match Call, an NPC/sign text box, an item pickup, and
	-- so on. All of them look the same to the nav script: steps fail, so it
	-- keeps bumping into a wall it can't see, burning replans until it gives
	-- up. This flag is how it tells "a text box is up" from "actually stuck".
	LOCK_FIELD_CONTROLS = 0x03000F2C,

	OBJECT_EVENT_SIZE = 0x24,
	OBJECT_EVENT_COUNT = 16,
	-- struct ObjectEvent
	OE_ACTIVE_BYTE = 0x00, -- bit 0 = active
	OE_GRAPHICS_ID = 0x05, -- which sprite: identifies the nurse
	OE_ELEVATION = 0x0B, -- low nibble = currentElevation
	OE_COORDS = 0x10, -- s16 x, s16 y
	OE_FACING = 0x18, -- low nibble = facingDirection
	OE_BEHAVIOR = 0x1E, -- currentMetatileBehavior

	-- struct PlayerAvatar
	PA_FLAGS = 0x00,
	PA_RUNNING_STATE = 0x02, -- 0 not moving, 1 turning, 2 moving
	PA_TILE_TRANSITION = 0x03, -- 0 settled on a tile, 1/2 mid-step
	PA_OBJECT_EVENT_ID = 0x05,

	-- struct MapHeader
	MH_MAP_LAYOUT = 0x00, -- const struct MapLayout *
	MH_EVENTS = 0x04, -- const struct MapEvents *
	MH_CONNECTIONS = 0x0C,
	-- struct MapLayout
	ML_PRIMARY_TILESET = 0x10,
	ML_SECONDARY_TILESET = 0x14,
	-- struct Tileset
	TS_METATILE_ATTRIBUTES = 0x10,
	-- struct MapEvents
	ME_WARP_COUNT = 0x01,
	ME_WARPS = 0x08, -- const struct WarpEvent *
	-- struct WarpEvent { s16 x, y; u8 elevation; u8 warpId; u8 mapNum; u8 mapGroup; }
	WARP_EVENT_SIZE = 0x08,
	WE_X = 0x00,
	WE_Y = 0x02,
	WE_WARP_ID = 0x05,
	WE_MAP_NUM = 0x06,
	WE_MAP_GROUP = 0x07,

	-- struct SaveBlock1 (via gSaveBlock1Ptr)
	SB1_POS = 0x00, -- struct Coords16
	SB1_LOCATION = 0x04, -- struct WarpData: where we are now
	SB1_LAST_HEAL_LOCATION = 0x1C, -- struct WarpData: whiteout / heal respawn
	-- struct WarpData { s8 mapGroup; s8 mapNum; s8 warpId; /*pad*/ s16 x, y; }
	WD_MAP_GROUP = 0x00,
	WD_MAP_NUM = 0x01,
	WD_WARP_ID = 0x02,
	WD_X = 0x04,
	WD_Y = 0x06,

	-- Map grid packing (include/global.fieldmap.h)
	MAPGRID_METATILE_ID_MASK = 0x03FF,
	MAPGRID_COLLISION_MASK = 0x0C00,
	MAPGRID_ELEVATION_MASK = 0xF000,
	MAPGRID_COLLISION_SHIFT = 10,
	MAPGRID_ELEVATION_SHIFT = 12,
	MAPGRID_UNDEFINED = 0x03FF,
	METATILE_ATTR_BEHAVIOR_MASK = 0x00FF,
	NUM_METATILES_IN_PRIMARY = 512,
	NUM_METATILES_TOTAL = 1024,

	ELEVATION_TRANSITION = 0,
	ELEVATION_MULTI_LEVEL = 15,

	-- Directions, as the game numbers them (DIR_* in global.fieldmap.h).
	DIR_SOUTH = 1,
	DIR_NORTH = 2,
	DIR_WEST = 3,
	DIR_EAST = 4,

	-- OBJ_EVENT_GFX_NURSE. The healing counter is found by looking for this
	-- sprite rather than by hardcoding coordinates, so it works in any
	-- Pokemon Center without knowing which map id that Center has.
	NURSE_GFX_ID = 58,
	-- The nurse stands behind a counter, so the player talks to her from two
	-- tiles away in a straight line rather than from an adjacent tile.
	NURSE_TALK_RANGE = 2,
}

-- Land wild-encounter tiles: behaviours with TILE_FLAG_HAS_ENCOUNTERS set in
-- sTileBitAttributes (src/metatile_behavior.c), minus the surfable ones, since
-- the agent walks rather than surfs. Using the game's own flag rather than
-- "tall grass" also covers long grass, ash, cave and desert encounters.
M.OW.ENCOUNTER_BEHAVIORS = {
	[0x02] = true, -- MB_TALL_GRASS
	[0x03] = true, -- MB_LONG_GRASS
	[0x05] = true, -- MB_UNUSED_05
	[0x06] = true, -- MB_DEEP_SAND
	[0x08] = true, -- MB_CAVE
	[0x0B] = true, -- MB_INDOOR_ENCOUNTER
	[0x23] = true, -- MB_ASHGRASS
	[0x24] = true, -- MB_FOOTPRINTS
}

-- Tiles that warp to another map when stepped on.
M.OW.WARP_BEHAVIORS = {
	[0x5F] = true, -- MB_NON_ANIMATED_DOOR
	[0x61] = true, -- MB_EAST_ARROW_WARP
	[0x62] = true, -- MB_WEST_ARROW_WARP
	[0x63] = true, -- MB_NORTH_ARROW_WARP
	[0x64] = true, -- MB_SOUTH_ARROW_WARP
	[0x68] = true, -- MB_ANIMATED_DOOR
	[0x69] = true, -- MB_UP_ESCALATOR
	[0x6A] = true, -- MB_DOWN_ESCALATOR
}

-- Ledge hops. Walkable, but a hop is a committed two-tile move that can't be
-- reversed, so the planner treats them as walls.
M.OW.LEDGE_BEHAVIORS = {
	[0x37] = true,
	[0x38] = true,
	[0x39] = true,
	[0x3A] = true,
}

-- Directional walls. The game checks these twice per step: the tile being
-- LEFT must not block movement outward, and the tile being ENTERED must not
-- block movement inward (IsMetatileDirectionallyImpassable). This is what
-- makes cliff edges one-way; planning with collision bits alone would route
-- the player off a cliff and strand them.
M.OW.BLOCKED_NORTH = { [0x31] = true, [0x33] = true, [0x34] = true, [0xBF] = true }
M.OW.BLOCKED_SOUTH = { [0x32] = true, [0x35] = true, [0x36] = true, [0xBF] = true }
M.OW.BLOCKED_EAST = { [0x2F] = true, [0x33] = true, [0x35] = true, [0xC0] = true, [0xBD] = true }
M.OW.BLOCKED_WEST = { [0x30] = true, [0x34] = true, [0x36] = true, [0xC0] = true, [0xBD] = true }

-- ---------------------------------------------------------------- memory map
-- Regions the address finder dumps (bus addresses, sizes in bytes).
M.RAM_REGIONS = {
	{ name = "ewram", start = 0x02000000, size = 0x40000 },
	{ name = "iwram", start = 0x03000000, size = 0x8000 },
}

return M
