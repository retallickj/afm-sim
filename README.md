# afm-sim
AFM simulation of SiDB structures.

# Hopping Model Animaator

## Keyboard shortcuts:
Key | Behaviour
:---: | ---
*Q* | Close the animator (and compile the recording if enabled)
*O* | Open the options panel
*S* | Timestamped screenshot of display window (no sidepanel) to local directory
*Space* | Advance the simulation one hop
*-* | Zoom out
*+* or *=* | Zoom in


## Mouse Commands
Button  | Behaviour
:---:   | ---
*Left Click*  | Clicking any hydrogen site will add/remove a fixed charge at that location. This will not automatically update the time before the next hop so you should advance the animation after any changes.
*Right Click* | Right clicking a DB will track its local potential on the options dialog. Re-click the same DB or a hydrogen site to end the tracking.

## Current Bugs

If there are ever fewer than 2 free DBs the cohopping implementation will throw an Error.
