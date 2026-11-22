-- Port selection for the bridge, kept apart from battle_bridge.lua so it can be tested without the emulator.
-- With several mGBA windows open (parallel training) each window takes the first free main port in the sequence
-- base, base+step, base+2*step ...; its address-finder (probe) port is main+1. The first window launched therefore
-- gets 8888/8889, the second 8890/8891, and so on, and training env i connects to port base + step*i.
local M = {}

-- bind(address, port) -> socket or nil, like mGBA's socket.bind. Returns the first socket that binds and its port.
function M.bindFirst(bind, base, step, count)
	for i = 0, count - 1 do
		local port = base + step * i
		local s = bind(nil, port)
		if s then
			return s, port
		end
	end
	return nil, nil
end

return M
