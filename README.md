# afm-sim
AFM simulation of SiDB structures.

# Hopping Model Animaator

## Keyboard shortcuts:
Key | Behaviour
--- | ---
*Q* | Close the animator (and compile the recording if enabled)
*O* | Open the options panel
*Space* | Advance the simulation one hop

Clicking any hydrogen site will add/remove a fixed charge at that location. This will not automatically update the time before the next hop so you should advance the animation after any changes.

## Current Bugs

If there are ever fewer than 2 free DBs the cohopping implementation will throw an Error.
