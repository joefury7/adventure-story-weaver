"""
Corpus Generator for Adventure Story Weaver
Assembles diverse, atmospheric public-domain and procedural adventure game narratives.
"""

import os
import random

PROLOGUES = [
    "You stand before the obsidian archway of the Sunken Citadel. A cold draft carries the scent of damp moss and aged parchment.",
    "The heavy iron doors of the Citadel of Torment creak open, revealing a spiral staircase winding into subterranean dark.",
    "You find yourself within the Whispering Catacombs beneath the ruins of Eldoria. Torches flicker along damp stone walls.",
    "A thick indigo fog clings to the roots of the Eldergrove. In the clearing ahead rests a moss-covered altar of forgotten kings.",
    "The sea spray batters against the barnacled hull of the derelict galleon trapped within the Black Coral Reef.",
    "Deep within the Smoldering Chasm, rivers of liquid fire illuminate crystalline pillars that stretch toward the vaulted cavern ceiling.",
    "You step into the Great Scriptorium of the Astral Observatory. Thousands of leather-bound grimoires float silently in the air.",
    "The frost-covered battlements of Winterhold overlook a jagged valley shrouded in eternal blizzard. An iron portcullis blocks the courtyard.",
    "You awaken on a bed of dry reeds inside a subterranean cave. The gentle trickling of mineral water echoes through the gloom.",
    "Before you lies the Gilded Tomb of Pharaoh Nephren-Ka. The golden hieroglyphics pulse with a faint, hypnotic crimson luminescence.",
]

ROOM_DESCRIPTIONS = [
    "The chamber is circular, carved from seamless volcanic glass. In the center rests a pedestal adorned with a silver skull whose eye sockets flicker with cerulean flame.",
    "Cobwebs drape like decaying curtains across the decaying wooden rafters. A desiccated chest bound in blackened bronze sits undisturbed in the northern corner.",
    "Ancient murals cover the crumbling limestone walls, depicting armored knights waging war against serpentine monstrosities that emerge from the sea.",
    "A crystal pool of subterranean water reflects the eerie glow of luminous fungi clinging to the damp cavern roof.",
    "Rows of ruined oak tables are scattered with shattered glassware, brass calipers, and yellowed alchemical notes detailing forbidden transmutations.",
    "A colossal statue of a winged gargoyle towers over a narrow marble bridge spanning an abyss of impenetrable shadow.",
    "Rusted chains dangle from the vaulted ceiling, their iron shackles swaying slightly despite the stagnant, suffocating air.",
    "The floor is paved with hexagonal tiles of jade and lapis lazuli, inscribed with concentric sigils of protective abjuration.",
    "A fireplace carved into the likeness of a roaring dragon mouth holds embers that refuse to turn to ash.",
    "Shelves of petrified wood line the walls, packed with clay jars containing preserved herbs, dried phoenix feathers, and bottled starlight.",
]

PLAYER_ACTIONS = [
    "You examine the intricate silver skull upon the pedestal.",
    "You draw your steel broadsword and tread cautiously across the creaking floorboards.",
    "You speak the command word etched into the pommel of your ancient blade.",
    "You light a fresh pitch-pine torch, casting leaping shadows across the damp carvings.",
    "You kneel beside the bubbling font of ethereal liquid and reach out with an iron flask.",
    "You search the desiccated remains of the fallen sentry for keys or journals.",
    "You pry open the corroded latch of the bronze chest with your hunting dagger.",
    "You trace your fingers along the glowing arcane sigils inscribed on the stone door.",
    "You ready your ash-wood recurve bow, nocking an arrow dipped in wyrm venom.",
    "You step carefully across the crumbling bridge, keeping your eyes on the chasm below.",
]

ENCOUNTERS_AND_EVENTS = [
    "Without warning, the silver skull rattles against the marble pedestal. A spectral voice whispers secrets of the celestial spheres, warning that only the pure of heart may claim the sapphire key.",
    "A grinding screech echoes as the flagstones beneath your boots shift. From the darkness above, a stone gargoyle snaps its stony wings and descends with razor talons outstretched.",
    "The bronze chest opens with a satisfying click. Inside lies a velvet-lined scabbard holding a dagger forged of star-iron, faintly humming with static electricity.",
    "The waters of the crystal pool ripple violently. A serpentine apparition emerges, its eyes glowing like twin emeralds as it assesses your resolve.",
    "A secret panel in the masonry slides back, revealing a narrow tunnel descending toward a forgotten armory filled with enchanted weaponry.",
    "The dormant embers in the dragon hearth flare into bright turquoise fire. The ambient chill vanishes, replaced by a radiant aura of warmth that invigorates your weary limbs.",
    "Ghostly figures materialize in the gloom—the remnants of the royal guard, bound by eternal oaths to defend the crown jewels from unworthy hands.",
    "A sudden gust of wind snuffs out your torch. In the absolute darkness, the faint luminescence of glowing footprints leads safely around a hidden pit trap.",
    "The ancient tome on the lectern flips open page by page of its own accord, pausing at an illustration depicting your exact face beneath a crowned halo.",
    "You feel a strange vibration resonating through the stone floor. Far below in the subterranean vaults, something ancient and massive shifts in its slumber.",
]

OUTCOMES_AND_EPILOGUES = [
    "With breath held, you pocket the artifact and slip into the shadows, leaving the tomb undisturbed once more.",
    "The monster collapses into a shower of harmless obsidian dust, leaving behind a glowing gem of wondrous radiance.",
    "You ascend the winding steps, emerging beneath a starlit sky as the mountain wind greets your triumphant return.",
    "The path forward is clear now. You tighten your grip on your gear and march deeper into the uncharted labyrinth.",
    "Having solved the riddle of the archons, the golden portal unlocks, bathing the hallway in welcoming amber light.",
    "You take refuge beside the hearth, binding your wounds as you prepare for the peril that awaits in the lower chambers.",
    "The ancient wards finally break, releasing the trapped souls of Eldoria to wander home across the astral plane.",
    "A sense of tranquil mastery washes over you as the magical resonance harmonizes with your inner spirit.",
]

def build_narrative_segment():
    prologue = random.choice(PROLOGUES)
    room = random.choice(ROOM_DESCRIPTIONS)
    action = random.choice(PLAYER_ACTIONS)
    event = random.choice(ENCOUNTERS_AND_EVENTS)
    outcome = random.choice(OUTCOMES_AND_EPILOGUES)
    
    return f"{prologue}\n{room}\n> {action}\n{event}\n{outcome}\n"

def generate_adventure_dataset(num_stories=1500, output_path="data/adventure_corpus.txt"):
    random.seed(42)
    stories = []
    for _ in range(num_stories):
        stories.append(build_narrative_segment())
    
    full_text = "\n---\n\n".join(stories)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    
    print(f"Generated adventure corpus: {len(stories)} stories, {len(full_text)} characters, {len(full_text.split())} words.")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    generate_adventure_dataset()
