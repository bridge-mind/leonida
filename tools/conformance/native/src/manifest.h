#pragma once

#include <filesystem>
#include <string>

#include "scenes.h"

// Writes manifest.json in the schema the diff tool consumes. Every check comes from
// scenes.h so the tool never parses C++. Holds no timestamps or run counts: two runs
// against the same adapter must produce identical bytes.
bool WriteManifest(const std::filesystem::path& path, const std::string& adapter_description,
                   const conform::Scene* scenes, int scene_count);
