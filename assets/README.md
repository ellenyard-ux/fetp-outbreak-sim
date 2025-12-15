Images for scenario scenes

# Village Photos

This directory contains photos for the village profiles in the simulation.

## Directory Structure

Photos should be organized by village name:
- `Nalu/` - Photos for Nalu village
- `Kabwe/` - Photos for Kabwe village
- `Tamu/` - Photos for Tamu village

## File Naming Convention

Photos should follow the naming pattern: `{village}_{number}_{description}.{ext}`

For example:
- `nalu_01_village_scene.png`
- `nalu_02_rice_paddies.png`
- `kabwe_01_mixed_farming.jpg`

## Supported Formats

- PNG (.png)
- JPEG (.jpg, .jpeg)

## Current Photos

### Nalu Village
- `nalu_01_village_scene.png` - General village scene
- `nalu_02_rice_paddies.png` - Rice paddies with standing water
- `nalu_03_pig_pens.png` - Pig cooperative facilities
- `nalu_04_health_center_exterior.png` - Health center building
- `nalu_05_market_day.png` - Market day scene

The simulation will automatically detect and display photos if they exist in the village directories. If no photos are found, it will fall back to SVG illustrations.
