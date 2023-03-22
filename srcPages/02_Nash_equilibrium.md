# [Best-response strategy and Nash equilibrium](@id nash_equilibrium)

In this chapter we define the concept of _best response_ and _Nash equilibrium_ and describe some methods to obtain the Nash equilibria of a given game.

```julia
# Let Julia activate a specific environment for this course instead of using the global one
using Pkg 
cd(@__DIR__)
Pkg.activate(".")
using LinearAlgebra, StrategicGames
```

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

!!! note
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
