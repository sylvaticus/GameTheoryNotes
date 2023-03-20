<!-- Part of student's notes on "Game Theory"
https://github.com/sylvaticus/GameTheoryNotes -->

# Basic concepts

In this unit we will introduce the main terminology, the concepts of Mixed strategies and equilibrium, in particular the Nash equilibrium.

```julia
# Let Julia activate a specific environment for this course instead of using the global one
using Pkg 
cd(@__DIR__)
Pkg.activate(".")
# And let's install the companion package "StrategicGames" from https://github.com/sylvaticus/StrategicGames.jl
Pkg.add(url="https://github.com/sylvaticus/StrategicGames.jl")
using LinearAlgebra, StrategicGames
```

## Basic definitions

**Utility** function:
- a mapping from a certain state of the world to a real number interpreted as measure of agent's happiness or satisfation
- it allows quantify preferences against different alternatives
- strictly speaking utility is an _ordinal_ measure, not a _cardinal_ one. When we apply an affine transformation (i.e. linear with eventually a constant, like $y = ax+c$) to the inputs of the utility function the _ranking_ of the preferences doesn't change. As well the comparition of utilities between different agents doesn't change (if the utility of agent 1 was higher than those of agent 2 and we apply an affine transformation to the inputs, e.g. we change the measure units, the utility of agent 1 remains higher than those of agent 2).

**Game** :
- the situation arising from the interaction of multiple utility maximising agents 

**Noncooperative** game:
- when the modelling unit is the (utility maximising) individual

**Coalitional** or **cooperative** games:
- when the modelling unit is the group

**Normal form game**:
- a game situation where each players play at the same time (no time dimension) and states of the word (utilities) depends only from the combined actions of all the players, without stochasticity (there could still be stochasticity in the choice of making decisions by the players)

**Bayesian game**:
- when the state of the world depends on stochasticity other than the players combined actions

**Extensive-form games**:
- include a timing dimension t that precises the _order_ of the actions taken by the various players
- it becomes relevant the degree of information that the agent knows at the times of making decisions  
 
A (_finite, N-person_) **normal-form** game is characterized by the following elements:
- $N$ is a finite (ordered) set of players indexed by $n$
- $A_n$ is the (ordered) set of actions available to player $n$. Each player can have different actions available (including in different number). The total possible states of the world correspond to all the possible actions that all the $N$ players can take and it is given by the _N_-dimensional tensor $A$ of size $(length(A_1),length(A_2),...,length(A_n)$. Note that as there isn't any stocasticity here, there is no distintion between a given set of actions and the resulting state of the world.
- $U$ is the utility levels associated to each corresponding state of the word, aka **pay-off matrix** ("tensor" would be a more appropriate word). This is a $N+1$ dimensional tensor of size $(length(A_1),length(A_2),...,length(A_n),N$. The last dimension has size $N$, as each player has its own utility functions of the various states of the world. Alternatively can be represented as a N dimensional tensor of tuples representing each the utility of the various players for the given state of the world.
  
We can further define:
- $S_n$ the (infinite) set of all the discrete probability functions that agent $n$ may want to use over its set of available actions $A_n$ in order to stocastically determine its action. Each individual probability distribution $s_{ni}$ is called **strategy**. Strategies $s_{ni}$ with a single action with non-zero probabilities are called **pure strategies**, while strategies with more than one available action assigned non-zero probabilities are called **mixed strategies** and strategies with non zero probabilities for all actions are named **fully mixed straegies**. The (sub)set of actions assigned non-zero probabilities is called **support** of the given strategy. We indicate with $s_{n}$ the strategy (PDF) emploied by player $n$ and with $s$ the **strategy profile**, the set of all particular strategies applied by all the individual players.
- $a_{length(A_1),length(A_2),...,length(A_n)}$ are the individual states of the world (the elements of the $A$ tensor) derived by the combined actions of all the $1,2,...,N$ players. These are also called an _action profile_.
- $E[U(s_{ni})]$ The _expected_ utility by player $n$ by employing strategy $s_{ni}$. Knowing (or assuming) the strategies emploied by the other players, the expected utility of a given strategy $i$ for player $n$ can be computed as $\sum_{a \in A} U_n(a) \prod_{j=1}^{N} s_j(a)$ that is it, for each state of the world $a$ we compute the probability (using the multiplication rule) that it arises using the strategies of all the players, _including the one specific we are evaluating ($i$ of player $n$)_, and we multiply it by the utility this state provides to player $n$ and we finally sum all this "possible" state to retrieve the expected value. 


### Examples

```julia
N = 3            # 3 players
A = [3,2,2]      # 3, 2,and 2 actions available for players 1,2,3 respectively
U = rand(A...,N) # pay-off tensor 
a = (2,1,2)      # one particular state of the world (action profile)
U[a...,1], U[a...,2], U[a...,3] # The utilities for the 3 players associated to this particular state of the world

s1 = [0.5,0.3,0.2] # One particular mixed-strategy emploied by player 1
s2 = [0,1]         # One particular pure-strategy emploied by player 2
s3 = [0.5,0.5]     # One particular mixed-strategy emploied by partner 3

s = [s1,s2,s3]     # A strategy profile

# Expected utilities for the 3 players under strategy profile s:
expected_utility = [sum(U[i1,i2,i3,n] * s[1][i1] * s[2][i2] * s[3][i3] for i1 in 1:A[1], i2 in 1:A[2], i3 in 1:A[3]) for n in 1:N]
expected_utility = StrategicGames.expected_payoff(U,s) # equivalent from the library package

```

### Particular types of games

**Common-playoff** games:
- a type of game where, for each action profile, all players derive the same utility 
- a "pure coordination" game: the players have no conflict interest and the only "difficulty" is for them to coordinate to get the maximum benefits

**Zero-sum** games (aka "**constant-sum**" games):
- a 2-players type of game where the sum of the utility for the 2 players for each action profile is constant
- a "pure competition" game, as if one action profile bring some additional advantage than the constant for the first player, this must be at the expenses of the other player 
- as games are insensible to affine transformations, this constant isn't restricted to be zero, it can be any constant value

**Prisoner-dilemma** games:
- a classical 2-actions, 2-players game with:
  -  a pay-off structure like:

  | p1 \ p2 | A     | B     |
  | ------- | ----- | ----- |
  | A       | a,a   | b,c   |
  | B       | c,b   | d,d   |

  where the first element in the tuple of each action profile (cell) represents the utility for player 1 (row player) and the second element the utility for player 2 (column player)
  - a numerical evaluation of $a,b,c,d$ such that $c > a$, $d > b$ and $a > d$, that is $c > a > d > b$

This game is partially cooperative and partially competitive. It can be seen that for each player, whatever the other player playes, it is always better to play `B` (for example for player 1, if player 2 plays `A`, $c$ is higher than $a$, and if player 2 plays `B`, $d$ is higher than $b$).
The action `B` then _dominates_ the other one for both the players even if they could coordinate they would be both better off by both choosing `A`.
In other words, as everyone behaves as _free rider_, the best feasible state is never reached. This is a frequent situation in economics in relation to the production of public goods: everyone wants to share the benefit from it, but nobody wants to bring the costs of their production.

The name derives from the original situation used to illustrate the case, of two prigioniers that can choose to deny their common crime they are accused (A) or confess to the authority (B), where if one alone confesses, he get free of prison ($c$) but the other get a harsh prison term ($b$); if both confess they get a standard prison term (d) and if they both deny they got a mild prison term ($a$). The outcome of the game is that it is rational for both of them to confess !

### Examples

A common pay-off game with 2 actions for the first player and 3 actions for the second player:
| p1 \ p2 | A     | B     | C    |
| ------- | ----- | ----- | ---- |
| A       | 1,1   |  2,2  | 3,3  |      
| B       | 4,4   |  5,5  | 6,6  |   

A constant-sum pay-off  game with 2 actions for the first player and 3 actions for the second player:
| p1 \ p2 | A     | B     | C     |
| ------- | ----- | ----- | ----- |
| A       | 1,3   | 5,2   | 3,1   |      
| B       | 4,0   | 5,-1  | 6,-2  |  

A prisoner-dilemma problem:
| p1 \ p2 | Deny  | Confess |
| ------- | ----- | ------- | 
| Deny    | -1,-1 |  -3,0   |       
| Confess |  0,-3 |  -2,-2  | 


## Best-response strategy and Nash equilibrium

Given a game in normal form and the (assumed) strategies of all players other than $n$, indicated as $s_{-n}$, the best-response strategy for player $n$ is:
$s^*_n$ such that $E[U(s^*_{n},s_{-n})] \ge E[U(s_{n},s_{-n})]  ~~ \forall ~~ s_{n} \in S_n $, that is the (possible mixed) strategy that results in the highest expected utility for player $n$ within all the strategies available to him.
A strategy is a **strictly best response** when it is unique within $S_n$, i.e. its expected utility, conditional to $s_{-n}$, is strictly larger tha any other $s_{ni}$. 

**Nash equilibrium**
The Nash equilibrium is simply the strategy profile(s) formed by only best-response strategies by all players.
If all these are strict best responses, we have a **strict Nash** equilibrium, otherwise we have a **weak Nash** equilibrium.

- under the Nash equilibrium each player maximise its expected utility conditional to the other players' action
- nobody that playes the Nash equilibrium has incentive to deviate from it
- if we have a strict Nash equilibrium, this is unique, otherwise may not (the other best response may not form a Nash equilibrium)
- if we allow players only for pure strategies a game may have none, one or multiple Nash equilibrium, but if we allow for mixed-strategies a game with finite number of players and actions will always have at least one Nash equilibrium
- if we allow players only for pure strategies a game may have strict or weak Nash equilibrium(a), but if we allow for mixed-strategies a game with always have weak Nash equilibrium(a)


## Pareto optimality of a state of the world

When we put ourself as an outside observer we would like a criteria to decide which outomes are better than others. Under the pareto optimal criteria (bear in mind that economists prefer to use the word "efficient" rather than "optimal") a given strategy profile is optimal if there could not be a player that could be made better off from the resulting (expected) state of the world without another player being made worst, that is there are no pareto-dominated strategy profile where at least one player could improve its expected utility without the other loose anything under another strategy profile.

Pareto efficient solutions are not unique. They form instead a "frontier" of the utilities that the game can bring to the players.

In the figure below the two axis refer to the expected utilities for players 1 and 2, red points ($a,b,c$) refer to Pareto dominated strategy profiles; green points $d,e,f$, leaning on the efficient frontier, are Pareto optimal strategies and grey points $g,h$ are simply points impossible to obtain in the given game.   

![pareto](imgs/pareto.png "Pareto frontier")

### Example

The following table provides the expected utility for a 3-players game, each with two pure actions, and show which of them are on the frontier on the last column:

| Str. profile | Player1 | Player 2 | Player 3 | Optimal |
| ------------ | ------- | -------- | -------- | ------- |
| A,D,F        | 6       | 4        | 10       |         |
| A,D,G        | 2       | 14       | 5        | *       |
| A,E,F        | 4       | 4        | 4        |         |
| A,E,G        | 8       | 8        | 8        | *       |
| B,D,F        | 4       | 12       | 5        | *       |
| B,D,G        | 7       | 8        | 1        |         |
| B,E,F        | 12      | 2        | 1        | *       |
| B,E,G        | 6       | 6        | 10       | *       |

We can see that this game has 5 Pareto-optimal strategy profiles. The $(A,D,F)$ strategy is instead dominated by the $(B,E,G)$ one, the $(A,E,F)$ strategy is dominated by the $(A,D,F)$, $(A,E,G)$, $(B,D,F)$ and $(B,E,G)$ ones and finally the $(B,D,G)$ strategy is dominated by the $(A,E,G)$ one.

The prisoner-dilemma is an interesting example of how the equilibrium outcome (confess, confess) is the only Pareto-dominated outcome of the game. 

## Finding the Nash equilibrium

We now convey the topic on how to find the Nash equilibrium. We already saw cases like the prisoner-dilemma where all players have a dominating strategy that doesn't depend from the other players' actions. This is the "easy" case, as the "confess" is the best-respons strategy for both and hence _(confess,confess)_ is indeed a Nash equilibrium. 

### 2-players game

Sometimes, when the game is small, we can look at the playoff matrix and directly apply the definition of Nash equilibrium to chechk if candidate strategies are Nash equilibrium.

#### Examples:

2-players common-payoff game: 
| p1 \ p2 | A     | B     | 
| ------- | ----- | ----- | 
| A       | 4,4   | 0,0   |      
| B       | 0,0   | 6,6   | 

It is easy to see that in this game both $(A,A)$ than $(B,B)$ represent a (weak) Nash equilibrium: given the other strategy, playing the same strategy is in both cases the (reciprocal) best response.
Let's try some mixed-strategies.

Is it $s_1 = s_2 = [0.5,0.5]$ a Nash equilibrium?

```julia
U      = [(4,4) (0,0); (0,0) (6,6)]
# This transform a n-players dimensional payoff tensor of tuples (like `U` in this case)
# to a n-players+1 dimensional tensor of scalars (where the additional dimension is relative to the various players)
payoff = expand_dimensions(U) 
s      = [[0.5,0.5],[0.5,0.5]]

expected_payoff(payoff,s)                     # (2.5,2.5)
expected_payoff(payoff,[[0.5,0.5],[0.6,0.4]]) # (2.4,2.4)
expected_payoff(payoff,[[0.5,0.5],[0.4,0.6]]) # (2.6, 2.6)
expected_payoff(payoff,[[1,0],[1,0]])         # (4, 4)   
expected_payoff(payoff,[[1,0],[0.8,0.2]])     # (3.2, 3.2)  
expected_payoff(payoff,[[0.8,0.2],[1,0]])     # (3.2, 3.2)
expected_payoff(payoff,[[0,1],[0,1]])         # (6, 6)   
expected_payoff(payoff,[[0,1],[0.2,0.8]])     # (4.8, 4.8)  
expected_payoff(payoff,[[0.2,0.8],[0,1]])     # (4.8, 4.8)
```
From the results above we notice that `(0.5,0.5)` for both players is NOT a Nash equilibrium, as each player has better respons strategies, while the two pure strategies `(1,0)` and `(0,1)` are.

Let't now take another example, the zero-sum game _Head or Tail_, where the first player wins if the 2 draws of the coin are the same, and player 2 win otherwise. It has the following payoff matrix:

| p1 \ p2 | H     | T     | 
| ------- | ----- | ----- | 
| H       | 1,-1  | -1,1  |      
| T       | -1,1  | 1,-1  | 

We can see that this game, contrary to the one before, doesn't have a pure strategy Nash equilibrium: if p1 plays H, p2 should plays T, at which point p1 should change its strategy to T, at which point p2 should change its strategy to H, and so on.
Let's see if instead `(0.5,0.5)` is a mixed-strategy Nash equilibrium.

```julia
U = [(1,-1) (-1,1); (-1,1) (1, -1)]
payoff = expand_dimensions(U)

# Player 2 improves payoff by deviating from same-action strategy:
expected_payoff(payoff,[[1,0],[1,0]])         # (1, -1)   
expected_payoff(payoff,[[1,0],[0.8,0.2]])     # (0.6, -0.6)  
expected_payoff(payoff,[[0,1],[0,1]])         # (1, -1)     
expected_payoff(payoff,[[0,1],[0.2,0.8]])     # (0.6, -0.6)

# Player 1 improves payoff by deviating from opposite-action strategy:
expected_payoff(payoff,[[1,0],[0,1]])         # (-1, 1)   
expected_payoff(payoff,[[0.8,0.2],[0,1]])     # (-0.6, 0.6)   
expected_payoff(payoff,[[0,1],[1,0]])         # (-1, 1)   
expected_payoff(payoff,[[0.2,0.8],[1,0]])     # (-0.6, 0.6)   

# Strategy profiles where one of strategy is (0.5,0.5) are (weak) Nash eq as nobody can improve its expectation changing its response strategy:
expected_payoff(payoff,[[0.5,0.5],[0.5,0.5]]) # (0, 0)
expected_payoff(payoff,[[0.5,0.5],[0.3,0.7]]) # (0, 0)
expected_payoff(payoff,[[0.5,0.5],[0.7,0.3]]) # (0, 0)
expected_payoff(payoff,[[0.3,0.7],[0.5,0.5]]) # (0, 0)
expected_payoff(payoff,[[0.7,0.3],[0.5,0.5]]) # (0, 0)

# Other mixed strategies doesn't seem to lead to an equilibrium:
expected_payoff(payoff,[[0.7,0.3],[0.7,0.3]]) # (0.16, -0.16)
expected_payoff(payoff,[[0.7,0.3],[0.6,0.4]]) # (0.08, -0.08)
```

#### Finding mixed strategies Nash equilibriums in 2 × 2 games
Let's now take a more comprehensive approach to find Nash equilibrium in playing `2 × 2` games, i.e. 2 players each with 2 possible actions.

The trick to find the equilibrium is to consider that each player must have a strategy that make the other one indifferent in terms of his actions, that is that conditional to the first player strategy, the expected utility of player 2 with respect to its own actions must be the same. Otherwise he would not play "at random".

Let's consider the following general payoff and strategies for a `2 × 2` game with mixed strategies:

| p1 \ p2 | A           | B           | 
| ------- | ----------- | ----------- | 
| A       | `(u1aa,u2aa)` | `(u1ab,u2ab)` |      
| B       | `(u1ba,u2ba)` | `(u1bb,u2bb)` |

`s1 = [p1a,1-p1a]`
`s2 = [p2a,1-p2a]`

**Finding p1a**:

Player 1 must find a strategy (i.e. `p1a`) such that:

`E[U]₂(payoff,s1,[1,0]) = E[U]₂(payoff,s1,[0,1])`

`u2aa * p1a + u2ba * (1-p1a) = u2ab * p1a + u2bb * (1-p1a)`

From which we find that:
`p1a = (u2bb-u2ba)/(u2aa-u2ba-u2ab+u2bb)`

**Finding p2a**:

Similarly player 2 must find a strategy (i.e. `p2a`) such that:

`E[U]₁(payoff,[1,0],s2) = E[U]₁(payoff,[0,1],s2)`

`u1aa * p2a + u1ab * (1-p2a) = u1ba * p2a + u1bb * (1-p2a)`

From which we find that:
`p2a = (u1bb-u1ab)/(u1aa-u1ab-u1ba+u1bb)`

Note that the computation of the equilibrium strategy for each player involves the _other_ player utility only.

#### Examples

```julia
function nash_2by2(payoff::Array{T,3}) where {T}
    size(payoff) == (2,2,2) || error("This function works only with 2 × 2 games")
    p1a = (payoff[2,2,2] - payoff[2,1,2]) / (payoff[1,1,2]-payoff[2,1,2] - payoff[1,2,2] + payoff[2,2,2])
    p2a = (payoff[2,2,1] - payoff[1,2,1]) / (payoff[1,1,1]-payoff[1,2,1] - payoff[2,1,1] + payoff[2,2,1])
    return [[p1a,1-p1a],[p2a,1-p2a]]
end

# Head or tail
U = [(1,-1) (-1,1); (-1,1) (1, -1)]
eq = nash_2by2(expand_dimensions(U)) # [[0.5, 0.5],[0.5,0.5]]

# A biased penalty kick game (kicker - the row player - is more efficient on B)
U = [(-1,1) (1,-1); (1,-1) (0, 0)]
eq = nash_2by2(expand_dimensions(U)) # [[0.33, 0.66],[0.33,0.66]]

# Battle of the sex
U = [(2,1) (0,0); (0,0) (1,2)]
eq = nash_2by2(expand_dimensions(U))  # [[0.66,0.33],[0.33,0.66]]
```

### Interpretation of mixed-strategies equilibrium

What does a mixed-strategy represents? Why should it be used ?

1. Confuse the opponent. In many competitive games (like the _Head or Tail_ one) apply a pure strategy would imply the other player be able to exploit to its own advantage. It is only by applying a random (i.e. mixed) strategy that the opponent can't exploit your strategy
2. Uncertainty over other players' strategies. Under this interpretation, a mixed strategy of a given player is the assessment of all other players concerning how likely are his pure strategies taken individually. Further, every action in the support of a mixed strategy in a Nash equilibrium is a best response to the player beliefs about the other players’ strategies. 
3. Empirical frequency. The entry of each action in a mixed strategy is the relative count on how often that action would be emploied in repeated games by the same players or with a different players selected at random from a hypothetical "player population"


### Multiple players game

Going beyond 2x2 games risks to become quickly intractable.

There are indeed no known algorithm that can _compute_ Nash Equilibrium in polynomial time with the size of the problem (action space), even more to answer questions like if there is a unique equilibrium, if there is a pareto efficient equilibrium, if an equilibrium whose expected payoff for player _n_ is at least _x_.... 
Neverthless, _verify_ if a given solution (an action profile) is a Nash Equilibrium, is relativelly computationally cheap, and some newest algorithms try to exploit this property.

Here we present two very different approaches. 

#### The Linear Complementarity formulation

The fist one is a Linear Complementarity formulation (LCP), that originally used a dedicated "solving" algorithm from Lemke-Howson (1964).
It's a mathematical programming problem that we can today solve with other methods (in `StrategicGames` we use the interior point method). Altougth it has a worst case exponential time with the size of the problem, it remains relativelly fasr in practice. The specific equilibrium that is retrieved depends from the initial conditions.

The LCP method finds the equilibrium conditions by exploiting a lot what a "game" is and the characteristics that a (Nash) equilibrium must have. In algeabric terms for a two-players game the problem corresponds to the following linear problem (notes: it seems more quadratic than linear actually due to the complementarity conditions):

(eq. 1) $~~\sum_{k \in A_2} u_1(a_1^j, a_2^k) * s_2^k + r_1^j = U_1^* ~~~~ \forall j \in A_1$
(eq. 2) $~~\sum_{j \in A_1} u_2(a_1^j, a_2^k) * s_1^j + r_2^k = U_2^* ~~~~ \forall k \in A_2$ 
(eq. 3) $~~\sum_{j \in A_1} s_1^j = 1, ~~ \sum_{k \in A_2} s_2^k = 1$
(eq. 4) $~~s_1^j \geq 0, ~ s_2^k \geq 0 ~~~~~~~~~~~~~~~~~~ \forall j \in A_1, \forall k \in A_2$
(eq. 5) $~~r_1^j \geq 0, ~ r_2^k \geq 0 ~~~~~~~~~~~~~~~~~~ \forall j \in A_1, \forall k \in A_2$
(eq. 6) $~~r_1^j * s_1^j =0, r_2^k * s_2^k =0 ~~~~~ \forall j \in A_1, \forall k \in A_2$

Where $u_1$ and $u_2$ are the $j \times k$ payoff matrices for the two players (a parameter here), while $s_1, s_2$ (the strategies for the two players), $U_1^*, U_2^*$ (the equilibrium expected utility for any action in the support of the two players) and $r_1, r_2$ are the decision variables of the problem (what we want to find).   

Equations 1 and 2 states that, for each of the two players, the expected utility for any possible action, given the strategies of the other player, must be constant, eventually less of a $r$ term, specific for that action and player.
The complementary conditions (eq. 6) guarantee that either this $r$ term is zero, or that action has zero probability of being selected by the given player (i.e. it is not in its strategy support).
Eq. 3 and 4 simply guarantee that $s$ are PMF (probability mass functions, i.e. discrete distributions, that is non-negative values that sum to 1).

We can formulate the problem in Julia as follow:

```julia
using JuMP, Ipopt # The first package is the algeabric language library, the second one is the interior point based solver engine

function nash_lcp2players(payoff,init=[fill(1/size(payoff,1), size(payoff,1)),fill(1/size(payoff,2), size(payoff,2))])    
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff tensor and the number of players")
    ndims(payoff) == 3 || error("This function works with only two players.")
    m = Model(Ipopt.Optimizer)
    set_optimizer_attribute(m, "print_level", 0)
    @variables m begin
        r1[j in 1:nActions[1] ] >= 0
        r2[j in 1:nActions[2] ] >= 0
        u[n in 1:nPlayers]
    end
    @variable(m, 0-eps() <= s1[j in 1:nActions[1] ] <= 1,  start=init[1][j])
    @variable(m, 0-eps() <= s2[j in 1:nActions[2] ] <= 1,  start=init[2][j])
    @constraints m begin
        slack1[j in 1:nActions[1]], # either rⱼ or sⱼ must be zero
            r1[j] * s1[j] == 0
        slack2[j in 1:nActions[2]], # either rⱼ or sⱼ must be zero
            r2[j] * s2[j] == 0
        utility1[j1 in 1:nActions[1] ], # the expected utility for each action must be constant, for each nPlayers
        sum( payoff[j1,j2,1] * s2[j2] for j2 in 1:nActions[2] ) + r1[j1]  == u[1]
        utility2[j2 in 1:nActions[2] ], # the expected utility for each action must be constant, for each nPlayers
        sum( payoff[j1,j2,2] * s1[j1] for j1 in 1:nActions[1] ) + r2[j2]  == u[2]
        probabilities1,
            sum(s1[j] for j in 1:nActions[1]) == 1
        probabilities2,
            sum(s2[j] for j in 1:nActions[2]) == 1
    end;
    @objective m Max u[1] + u[2]
    optimize!(m)
    #print(m) # if we want to print the model
    status = termination_status(m)
    optStrategies = Vector{Vector{Float64}}()
    optU          = Float64[]
    if (status == MOI.OPTIMAL || status == MOI.LOCALLY_SOLVED || status == MOI.TIME_LIMIT) && has_values(m)
        optStrategies1 = value.(s1)
        optStrategies2 = value.(s2)
        optStrategies = [optStrategies1,optStrategies2]
        optU = value.(u)
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU)
end
```

We can try it with the same problems we "solved" analytically before:

```julia
# Head or tail
U = [(1,-1) (-1,1); (-1,1) (1, -1)]
eq = nash_lcp2players(expand_dimensions(U))
eq_strategies = eq.equilibrium_strategies # [[0.5, 0.5],[0.5,0.5]]

# A biased penalty kick game (kicker - the row player - is more efficient on B)
U = [(-1,1) (1,-1); (1,-1) (0, 0)]
eq = nash_lcp2players(expand_dimensions(U))
eq_strategies = eq.equilibrium_strategies # [[0.33, 0.66],[0.33,0.66]]

# Battle of the sex
U = [(2,1) (0,0); (0,0) (1,2)]
eq = nash_lcp2players(expand_dimensions(U)) 
eq_strategies = eq.equilibrium_strategies # [[0.66,0.33],[0.33,0.66]]

# A 2-players game with 2 and 3 actions respectively
U = [(1,-1) (-1,1) (1,0); (-1,1) (1, -1) (0,1)]
eq = nash_lcp2players(expand_dimensions(U)) 
eq_strategies = eq.equilibrium_strategies # [[0.66, 0.33],[0, 0.33, 0.66]]
```

!!! Warning
    If you run the code above you will get, due to the computational approximations, sligh different results, e.g. `1.7494232335004063e-20` instead of `0` in the last example

The `StrategicGames` package provides the `nash_lcp(payoff;init)` function, a generalisation of the algorithm above to `n` players, where each player can have a different actions space.

For example:

```julia
# This example is taken from https://www.youtube.com/watch?v=bKrwQKUT0v8 where it is analytically solved
U = [(0,0,0) ; (3,3,3) ;; (3,3,3) ; (2,2,4) ;;;
     (3,3,3) ; (2,4,2) ;; (4,2,2) ; (1,1,1) ;;;]
eq = nash_lcp(expand_dimensions(U))
eq_strategies = eq.equilibrium_strategies
p = -1 + sqrt(10)/2 # approximatively 0.5811
eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]] # true
```

#### The Support Enumeration Method
