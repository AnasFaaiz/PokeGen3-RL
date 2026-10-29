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

-- ---------------------------------------------------------------- memory map
-- Regions the address finder dumps (bus addresses, sizes in bytes).
M.RAM_REGIONS = {
	{ name = "ewram", start = 0x02000000, size = 0x40000 },
	{ name = "iwram", start = 0x03000000, size = 0x8000 },
}

return M
