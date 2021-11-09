# Translating OpenAi Five to Divinity Original Sin 2 (and possibly to Baldur’s Gate 3)





## Motivation
### Introduction
My main motivation for this project is pure curiosity. It has been some time since I dipped my toes into Reinforcement Learning (RL), and now I have finally read the OpenAI Five paper, my motivation is at an all time high.
However, I also want to build something actually useful. To keep me focused on building something of real value, I will aim to stick to two key principles;


* Build something that improves the player experience; and
* What I build needs to be comparable to how humans play. In essence, the bots that I build must play by the same rules humans do.


The second point here is very handy in pushing me in the right direction when considering technical implementations.


### How does combatAI currently work?
To give some context to how a RL approach could benefit the player experience, we can look into how things currently work and see where we could improve. Taken from https://docs.larian.game/Combat_AI we have the following algorithm outline:


```Let's say the Ai has the Projectile_Fireball skill and go through the steps that the Ai takes to calculate the scores for possible actions:
1. Check first: can we cast this skill? Skill conditions, cooldowns, difficulty mode, statuses preventing us from casting, etc. No need to calculate anything if we can't even cast.

2. Find interesting targets to cast the skill on. Ai checks for characters, items (barrels could explode), and positions on the terrain. We run a simulation on each of these targets to see what effect it will have. This simulation on every target gives us damage scores, health scores, buff scores, etc. for every object we hit (the ActionScore).

3. Calculate the 'final' score of every Action (at this point this is purely the skill being casted). The different scores are thrown together, affected by a whole bunch of target specific modifiers, and balanced so that we end up with one single value for the score.

4. Only the interesting Actions, based on the 'final' score in the previous step, will be investigated further. We start to find a good position to cast from. The character might not be able to cast from it's current position (out of sight, out of range, etc.), or the character might not like its current position due to nearby enemies. PositionScores will be calculated and we hope to find a good position for every action.

5. After finding a good position we start calculating the MovementScore. Does the character need to walk through a fire surface? Could we jump instead of walking? Do I have enough AP to even walk there this turn?

6. In the end only one Action will be on top. We have a skill (or default attack), a target, a position, and even the path has been checked. The action will be fetched from script and executed.

7. What if we don't have any Action that is deemed positive? That's when the fallback action kicks in. The fallback action tries to find a good position for the character and walks there.
```
After reviewing this, we can make the following observations:
1. It appears movement occurs all at once, meaning that the bot may leave themselves exposed after casting a skill.
2. It appears that a lot of calculations occur to determine the optimal action. It would be interesting to compare the compute time between this and an RL based approach.
3. The bots appear to be greedy. By this I mean that there doesn’t appear to be any consideration for the consequence of the action taken other than the immediate effect.


## Technical Details
Designing the Observation Space
An observation space is a narrowed down game state that each bot has access to. A good observation space must have the following two properties:
1. It must hold sufficient information such that a bot can make meaningful decisions; and
2. It must not utilise any information that a human player does not have access to. One of the main motivations for this work is to build a bot that plays by the same rules as a human player, so exposing hidden information to the bot via the observation space violates this constraint.


We cannot quantify how useful some information will be when we first begin, and what we deem to be adequate will probably evolve over the lifetime of the project. However, we can probably get a good idea if we simply note what human players use to make decisions. The table below is an incomplete breakdown of what I believe would be a good baseline for our bots’ observation space.





| Observation Category | Observation | Related Osiris Queries/Calls/Events | Implementation Notes |
|----------------------|-------------|-----------------------------------|----------------------|
| Overall Combat       | Units in combat| Osiris/API/IterateParty | May need to extend to all units in combat |
| | Potential combat units | | |
| Unit Statuses [1] | Is ally? | Osiris/API/CharacterIsAlly Osiris/API/CharacterIsPartyMember | |
| | Is hostile? | Osiris/API/CharacterIsEnemy Osiris/API/CharacterIsInFightMode | | 
| | Is neutral? | Osiris/API/CharacterIsNeutral Osiris/API/CharacterCanFight | | 
| | Is Summon? | Osiris/API/CharacterIsSummon | |
| | Max HP| | |
| | Current HP| Osiris/API/CharacterGetHitpointsPercentage| |
| | Max AP | | |
| | Current AP | | |
| | Available Statuses | Osiris/API/HasActiveStatus Osiris/API/HasAppliedStatus | Will have to write an iterator to get all statuses |
| | Status Duration | Osiris/API/GetStatusTurns | |
| | Position (Global and relative to current bot) | Osiris/API/GetPosition Osiris/API/GetRotation Osiris/API/GetAngleTo | |
| | Defences (Physical Armour and Magical Armour) | Osiris/API/CharacterGetArmorPercentage Osiris/API/CharacterGetMagicArmorPercentage | |
| | Unit Level | Osiris/API/CharacterGetLevel | |
| Unit Surroundings (These should be found per unit in combat)| Surface Type | Osiris/API/GetSurfaceGroundAt (73 types) | |
| | Distance to other units | Osiris/API/GetDistanceTo | |
| | Has line of sight | Osiris/API/HasLineOfSight | |
| | Destructible Items | Osiris/API/ItemIsDestructible | |


## Designing the Action Space
The action space is simply a representation of the actions a bot can take, such as moving position, attacking a target, using a consumable, etc. These actions are implemented using Osiris Calls (and possibly Procedures for more complex actions?). The following is an incomplete breakdown of how to group these actions:
* Movement
   * Needed parameters would include:
      * Target (as an x, y, z 3-vector)
      * Sneaking (Boolean)
      * Sprinting (Boolean)
* Attacking with weapon
   * Needed parameters would include:
      * Target (Area or Unit)
         * If Area, an x, y, z 3-vector would be needed.
         * If unit, which unit?
* Skill Usage
   * Needed parameters would include:
      * Which spell?
      * Target (Area or Unit)
         * If Area, an x, y, z 3-vector would be needed.
         * If unit, which unit?
However, there needs to exist a translation layer between the ML model and the Osiris engine, which will be responsible for converting a vector to a meaningful Osiris Call.


## Designing the Reward Function
The reward function is responsible for giving valuable feedback to the bot in order for it to improve in successive timesteps. In most combat scenarios [2], the objective is to eliminate the other party. Whilst this objective is true, it is also very broad, and leads to a common obstacle in RL called the sparse reward problem. What this basically means, is that throughout the whole combat, the only time the bot will get a reward is at the end. This, in turn, makes it extremely slow to train our bot, if not outright impossible, as it is highly unlikely our bot will make enough random guesses initially to win the combat.
To overcome this, a common solution is to introduce Reward Shaping. This is the procedure of breaking our objective into micro-objectives in order to provide guidance to our bot. So, how do we do this? As with when we were designing our observation space, let's put ourselves in a player’s perspective. The following is an incomplete list, breaking down the reward function:
* Reducing the enemy team numbers.
   * Killing enemies; and
   * Not attacking neutrals.
* Reduce the actions the enemy can take (maybe optimise this for passive bots?)
* Debuffs:
   * Increase the number of debuffs on enemies.
   * Decrease the number of debuffs on allies.
* Buffs:
   * Increase the number of buffs on allies.
   * Decrease the number of buffs on enemies..
* Maintaining HP
   * For example, casting healing spells.
* Positioning
   * Having highground gives bonuses to attack roles.
   * Having cover reduces chances of being hit.
   * Certain locations may be riskier than others (e.g. next to interactables)
* Spending AP
   * Using all available resources is most likely better than not doing so, except for the circumstance where carrying over AP into the next turn would be beneficial.
All of the above will most likely contribute in different amounts. Therefore it is anticipated that some time would be dedicated to tuning the weights of these factors.


## Choosing a Model Architecture
## System Architecture
## Difficulties and Novel Contributions
This section explains the potential difficulties one might face when going about implementing this.
When to obtain the gamestate?
Unlike DOTA2, DOS2 and BG3 use a turn based system as opposed to realtime. It does not make sense to retrieve a gamestate at fixed-time intervals, as there is a chance that either:
1. Nothing has changed since the last state.
2. Something changed in a frame that the bot didn’t see.


Therefore, when do we query for a gamestate? Below is a scribble of thoughts I have on this:


```
- Fixed time interval? - Seems odd as its turn based.

- Every time an action is used? - Could work?

- Just use every gamestate update? - Would work in small combat focused environments, but if non-combat actions are happening somewhere else in the world, do we really want to respond to this? Does it actually affect the combat in any way?

- (For DOS2) Every time an AP is spent? (For BG3) Everytime an action, bonus action or portion of movement is spent? - Seems okay, but what if an external event happens which affects combat? For example, when a new character enters combat? No AP has been spent, but it is well worth considering this new character in our decision making process.

- Maybe we make a custom CombatState, and we receive updates whenever this changes? This seems to be the best answer as it most closely resembles human playing?
   - Okay but how do we do this? This requires some more knowledge of what we can query.
   - However, isn't this just if our observation has changed, therefore, act? What are the limitations here? One example is if our observation is measuring distance to other units, our observation will be continuously changing.
      - Maybe only update positions on a significant change, e.g. amount of movement for a single AP usage?
```

## Can bots learn across multiple maps?
DOTA is always played on the same map (at least when training on a single patch), completely negating the overhead the bots would need to learn various maps. DOS2 and BG3 combat can happen anywhere, therefore our system would need to be able to handle that. One solution would be to force combat to happen in specific regions of a level, but that directly violates one of our original motivations; improving player experience. Locking combat to certain regions removes the freedom a player has, so this is not really a solution.
The only other solution is to train our bots across various maps and discover if what they learn transfers to unseen maps. I believe this has already been proven in the case for DOTA, but not directly. One feature of the observation space that OpenAI Five uses is the immediate surroundings of all units present on the map [3]. These surroundings are changing every frame the bots receive, and are not constrained in any way. Therefore, why is our set-up any different? From the bots perspective in OpenAI Five, they have no knowledge of the map boundaries, so why should ours need them?
In short, I believe the bots we would build would have the capacity to learn various maps, as the features from the observation space are somewhat map independent anyway.

## Can bots adjust to dynamic team sizes?
Combat in DOS2 and BG3 can involve varying team sizes, and even sometimes more than two parties (allies, enemies and neutrals). As with the multiple maps discussion above, I do not think this is an issue. Neither team size nor number of teams are directly hard-coded into our observation space. Whilst it does include which units are in combat, there is no notion of how many there should be. The same thing goes for neutral units.


## How do we model different bot behaviour?
As mentioned previously, combat need not end in death. One way to change the behaviour would be to simply weight the reward function differently for the desired behaviour. Let's say we have an enemy team of evil cultists that will do anything to kill the party. In order to reflect this behaviour, let's weight the part of the function that prioritizes killing more. What about a bunch of paladins and clerics who gain their power from a lawful good deity? Weight the part of the function that prioritises maintaining ally HP, i.e. favour casting healing spells.


What about difficulty levels? Rather than just increasing or decreasing AC or HP, let's just stop the training early. This allows us to abide by one of our motivations; building bots that play by the same rules as players.




## Overall Plan
## Closing Thoughts








________________
[1] Quite possible to use https://github.com/Norbyte/ositools/blob/master/LuaAPIDocs.md#character-stats as a way to get all of the information returned in a single query.

[2] Not all combat in RPGs must end in death! Well, it might for someone doing an evil playthrough, but there may well be times that it makes narrative sense for a party to be disarmed, or for the combat to diffuse to dialog.

[3] Technically, not all units, but 189 of the total units, which appears to suffice quite well considering the results.