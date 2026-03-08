import openroad as ord
import torch

class CongestionMap:
    def __init__(self, design=None, device=None, dtype=torch.float32, init_grt=True):
        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.capacity_map, self.x_min, self.x_max, self.y_min, self.y_max = getGridXY(
            design, device=self.device, dtype=self.dtype, init_grt=init_grt
        )
        tile_size_x = (self.x_max - self.x_min) / (self.capacity_map.shape[0] - 1) if self.capacity_map.shape[0] > 1 else 1
        tile_size_y = (self.y_max - self.y_min) / (self.capacity_map.shape[1] - 1) if self.capacity_map.shape[1] > 1 else 1
        self.tile_size = (tile_size_x, tile_size_y)
        self.usage_map = torch.zeros_like(self.capacity_map, dtype=self.dtype, device=self.device)

        

def getGridXY(design=None, device=None, dtype=torch.float32, init_grt=True):
    """
    Return (capacity_map, x_min, x_max, y_min, y_max) from the GCell grid.
    """
    if device is None:
        device = torch.device("cpu")
    if design is None:
        db = ord.get_db()
        if db is None:
            raise RuntimeError("OpenROAD database is not initialized.")
        chip = db.getChip()
        if chip is None:
            raise RuntimeError("OpenROAD database has no chip loaded.")
        block = chip.getBlock()
    else:
        db = ord.get_db()
        block = design.getBlock()
    design.evalTclString("set sig_layers \"M2-M9\"")
    design.evalTclString("set clk_layers \"M2-M9\"")
    design.evalTclString("set_routing_layers -signal $sig_layers -clock $clk_layers")
    if block is None:
        raise RuntimeError("OpenROAD block is not available.")

    gcellgrid = block.getGCellGrid()
    if gcellgrid is None and init_grt and design is not None:
        # GRT preprocessing only (no routing)
        design.getGlobalRouter().initCongestionMap()
        gcellgrid = block.getGCellGrid()

    if gcellgrid is None:
        # fallback synthetic grid (capacity unknown -> 0)
        die = block.getDieArea()
        tile = block.getGCellTileSize()
        x_count = max(1, (die.xMax() - die.xMin()) // tile)
        y_count = max(1, (die.yMax() - die.yMin()) // tile)
        grid_x = [die.xMin() + i * tile for i in range(x_count)]
        grid_y = [die.yMin() + i * tile for i in range(y_count)]
        capacity_map = torch.zeros((x_count, y_count), dtype=dtype, device=device)
        return capacity_map, grid_x[0], grid_x[-1], grid_y[0], grid_y[-1]

    grid_x = list(gcellgrid.getGridX())
    grid_y = list(gcellgrid.getGridY())
    if not grid_x or not grid_y:
        empty_map = torch.empty((0, 0), dtype=dtype, device=device)
        return empty_map, None, None, None, None

    x_min = min(grid_x)
    x_max = max(grid_x)
    y_min = min(grid_y)
    y_max = max(grid_y)

    if db is None:
        raise RuntimeError("OpenROAD database is not initialized.")
    tech = db.getTech()
    if tech is None:
        raise RuntimeError("OpenROAD tech is not available.")

    layers = [layer for layer in tech.getLayers() if layer.getRoutingLevel() > 0]
    x_count = len(grid_x)
    y_count = len(grid_y)
    capacity_map = torch.zeros((x_count, y_count), dtype=dtype, device=device)
    for x in range(x_count):
        for y in range(y_count):
            capacity_value = 0.0
            for layer in layers:
                capacity_value += gcellgrid.getCapacity(layer, x, y)
            capacity_map[x][y] = capacity_value

    print(f"capacity_map.shape: {capacity_map.shape}")
    return capacity_map, x_min, x_max, y_min, y_max
