#include <Simulation/Simulation.h>

#ifdef DEPRECATED

void Simulation::cpuReallocateGrid(int32 threadId, Cells<std::atomic<int32>> cellCounts, Grid grid, Particle** memoryBuffer, std::atomic<int32>& blockPointer) {

	if(threadId >= config::simulation::boundingBox::nCells) {
		return;
	}

	auto cellIndex = Simulation::map1dIndexTo3dCell(threadId);

	auto& cell = grid[cellIndex.z][cellIndex.y][cellIndex.x];

	int32 const cellCount = cellCounts[cellIndex.z][cellIndex.y][cellIndex.x];

	cell.Size = cellCount;
	cell.FreeIndex = 0;

	int32 blockBegin = blockPointer.fetch_add(cellCount);
	int32 blockEnd = blockBegin + cellCount;

	// zero the memory for safety
	for(int32 i = blockBegin; i < blockEnd; ++i) {
		memoryBuffer[i] = nullptr;
	}

	cell.Particles = memoryBuffer + blockBegin;

}

#endif