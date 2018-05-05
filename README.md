# afm-sim (HoppingDynamics engine in SiQAD)
Hopping dynamics simulation and animation for silicon dangling bond devices.

# Hopping Model Animator

Interactive simulation of the surface charge configurations. Charge dynamics are
simulated using Hopping rate calculation from a choice of Hopping Models.

## Main Functionality

* Approximately real time simulation of surface configurations.
* Control of certain physical and modeling parameters (see **Options Panel** below).
* Automatic population control through surface-bulk(reservoir) hopping.
* Placement of doubly occupied perturbers during simulation.
* Integration of a simple STM tip model (**In Progress**)
* Clocking field generation.

## Keyboard shortcuts:
Key | Behaviour
:---: | ---
**Q** | Close the animator (and compile the recording if enabled)
**O** | Open the options panel
**P** | Pause animation
**T** | When clocking, start a timer which automatically pauses the animation after a quarter phase.
**S** | Timestamped .png screenshot of display window (no sidepanel) to local directory
**Shift + S** | Timestamped .svg/.pdf screenshot of display window with capture presets.
**Space** | Advance the simulation one time step, somewhat deprecated.
**-** | Zoom out
**+** or **=** | Zoom in
**E** | Zoom extents
**L** | Start line-scan at current dimer row if tip channel included.
**D** | Detach GUI hook for Debug


## Mouse Commands
Button  | Behaviour
:---:   | ---
**Left Click**  | Clicking any hydrogen site will add/remove a fixed charge at that location. This will not automatically update the time before the next hop so you should advance the animation after any changes.
**Right Click** | Right clicking a DB will track its local potential on the options dialog. Re-click the same DB or a hydrogen site to end the tracking.
**Middle Click** | Panning
**Control + Wheel** | Zoom anchored at cursor.

## Options Panel

The options panel offers a selection of parameter controls/views for debugging purposes. Details of slider controls can be found in the hover tooltips.

|Field   |   Context
| :---:   | ---
|**DB-Beff**     | Local potential experience by targetted DB, see **Right Click** behaviour
|**Lifetime**    | Countdown for tracked DB. When the countdown runs out, the occupying charge will hop.
|**\# electrons**| Current number of electrons in the surface DBs.
|**Runtime Load**| Percentage of time between frames used for simulating the surface state. If greater than 100%, the animation is no longer running in real time.
|| <p></p><big>**Animation Controls**</big>|
|**Viewer**   | See below
|**log(rate)**| Changes the time step of the animator by 10^rate
|| <p></p><big>**Hopping Model**</big>|
|**lambda**   | Self trapping energy. At lambda=0, hopping models are calibrated to match the 1-2-2-1 results.
|**factor**   | Hopping rates are multiplied by 10^factor. Use instead of **rate** to increase just the hopping rates.
|**FRH**      | To increase performance, hops are only allowed within a certain range and cohopping only allowed from occupied DB pairs within a certain range. The FRH parameters change this range.
|| <p></p><big>**Bulk Properties**</big>|
|**mu**          | Chemical potential for utomatic population mechanism. This is effectively the energy difference between the Fermi level and the DB- state of an isolated DB.
|| <p></p><big>**Tip Properties**</big>|
|**Tip**    | If the Tip channel is included in the Hopping model, "Tip properties" and "Tip programs" will be available in the options
|**scale**  | Scales the tip influence, TIBB and ICIBB. Tip seems to be fairly well behaved for scale < 0.3 or so.
|**epsr**   | Relative permittivity for Coulomb interactions between DBs and image charges. Lower epsr will make ICIBB stronger
|**H** | Tip height above the surface, in pm
|**ICIBB R**  | Tip radius for ICIBB (image charge localised near the point of the tip)
|**TIBB R**   | Tip radius for TIBB (blunt tip with long range contact potential influence)
|**rate**     | Tip scanning speed
|| <p></p><big>**Tip Programs**</big>|
|**Padding**  | Extra space added to the outside of line and full (2D raster) scans
|**Lines**    | Number of lines/rows in the full scan
|**Line**   | Start a line scan horizontally along the nearest row of atoms.
|**Full**   | Scan the tip over a region containing the full device.
|| <p></p><big>**Clocking Field**</big>|
|**Clocking** | If the Clocking channel is incuded in the Hopping model, "Clocking Field" will be available in the options. The clock is a sinusoidal potential that travels to the right and effectively shifts the DB- level.
|**Frequency**  | Frequency of the clocking field
|**Amplitude**  | Amplitude of the clocking field, in eV
|**Offset**     | Offset of the clocking field, essentially equivalent to more **mu**
|**Flat Clock** | Currently broken: essentially sets the wavelength to infinite.


## Viewer

![alt text][viewer]

The Viewer is an additional tool which shows a representation of the device (currently collapsed onto the x axis). A well is drawn for each well with the black line indicating the energy of the DB- state and the green dots showing the current charge locations. For significant **lambda**, the dots will be noticeably lower than the DB- levels to indicate the self-trapping. If a DB is being tracked (see **Right Click** behaviour), blue lines will be shown for the possible hopping targets which indicate the effective energy level seen by the tracked charge. The tip will be shown if included with both radii shown and the apex indicated. *Clearly insert figure here*.

## Current Bugs

If there are ever fewer than 2 free DBs the cohopping implementation will throw an Error.
*Edit: 2018.05.02 Not sure if this still happens*


[viewer]: img/viewer.png "Viewer Example"
