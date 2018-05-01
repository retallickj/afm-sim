# afm-sim
AFM simulation of SiDB structures.

# Hopping Model Animator

## Keyboard shortcuts:
Key | Behaviour
:---: | ---
**Q** | Close the animator (and compile the recording if enabled)
**O** | Open the options panel
**P** | Pause animation
**T** | When clocking, start a timer which automatically pauses the animation after a quarter phase.
**S** | Timestamped .png screenshot of display window (no sidepanel) to local directory
**Shift + S** | Timestamped .svg screenshot of display window with capture presets.
**Space** | Advance the simulation one hop
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

Field   |   Context
:---:   | ---
DB-Beff     | Local potential experience by targetted DB, see *Right Click* behaviour
\# electrons| Current number of electrons in the surface DBs.
mu          | Chemical potential of the DBs w.r.t. the bulk for automatic population mechanism
log(rate)   | Animation speed control: the time between hopping events is sped up by a factor of 10^x for log(rate) x. <br><br> *Note: This is excluding computation time for the HoppingModel so there is no guarantee at this point of real time operation.*


## Current Bugs

If there are ever fewer than 2 free DBs the cohopping implementation will throw an Error.
