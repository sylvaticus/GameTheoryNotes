# [Best-response strategy and Nash equilibrium](@id nash_equilibrium)

In this chapter we define the concept of _best response_ and _Nash equilibrium_ and describe some methods to obtain the Nash equilibria of a given game.

```julia
# Let Julia activate a specific environment for this course instead of using the global one
using Pkg 
cd(@__DIR__)
Pkg.activate(".")
using LinearAlgebra, StrategicGames
using Combinatorics, JuMP, Ipopt, HiGHS
```

Given a game in normal form and the (assumed) strategies of all players other than $n$, indicated as $s_{-n}$, the best-response strategy for player $n$ is:
$s^*_n$ such that $E[U(s^*_{n},s_{-n})] \ge E[U(s_{n},s_{-n})]  ~~ \forall ~~ s_{n} \in S_n $, that is the (possible mixed) strategy that results in the highest expected utility for player $n$ within all the strategies available to him.
A strategy is a **strictly best response** when it is unique within $S_n$, i.e. its expected utility, conditional to $s_{-n}$, is strictly larger tha any other $s_{ni}$. 

!!! warning
    Computationally, due to approximations in computations with float numbers, it is not possible to numerically distinguish between a strict and a weck best response

Finding the _best response_ for a single player is computationally relativeelly cheap, as it involves solving a linear optimisation problem where we find the strategy that maximises the player's expected payoff subject to the strategy being a discrete probability distribution (i.e. all entries non negative and summing to 1).

The problem is implemented in the `StrategyGames.best_response(payoff,strategy_profile,player)` function.
In the example below we want to find the best response for player 2 in the prisoner's dilemma game when the player 1 plays to deny the crime. As you can see the best response is to betray to player 1 and confess.

```julia
payoff = [(-1,-1) (-3,  0);
          ( 0,-3) (-2, -2)] # prisoner dilemma (strategy 1 is to deny the crime, 2 is to confess it)
# This transform a n-players dimensional payoff array of tuples (like `U` in this case)
# to a n-players+1 dimensional array of scalars (where the additional dimension is relative to the various players)
payoff_array = expand_dimensions(payoff);
s1 = [1,0] # player one's strategy is to always deny
s2 = [1,0] # these are the initial values for player two's strategy (what we are looking for) in the underlying optimisation problem
best_response(payoff_array,[s1,s2],2) # (0,1)
```
We can also check if a particular strategy is a best response one with `is_best_response(payoff_array,strategy_profile,nplayer)`:

```julia
is_best_response(payoff_array,[s1,s2],2)    # false
is_best_response(payoff_array,[s1,[0,1]],2) # true
```

**Nash equilibrium**
The Nash equilibrium is simply the strategy profile(s) formed by only best-response strategies by all players.
If all these are strict best responses, we have a **strict Nash** equilibrium, otherwise we have a **weak Nash** equilibrium.

- under the Nash equilibrium each player maximise its expected utility conditional to the other players' action
- nobody that playes the Nash equilibrium has incentive to deviate from it
- if we have a strict Nash equilibrium, this is unique, otherwise may not (the other best response may not form a Nash equilibrium)
- if we allow players only for pure strategies a game may have none, one or multiple Nash equilibrium, but if we allow for mixed-strategies a game with finite number of players and actions will always have at least one Nash equilibrium
- if we allow players only for pure strategies a game may have strict or weak Nash equilibrium(a), but if we allow for mixed-strategies a game with always have weak Nash equilibrium(a)

We can check if a particular strategy profile is a Nash equilibrium with `is_nash(payoff_array,strategy_profile)`.
This simply check if `is_best_response` is true for all players:

```julia
is_nash(payoff_array, [s1,s2])       # false
is_nash(payoff_array, [[0,1],[0,1]]) # true - both players always defect
```

## Finding the Nash equilibrium

We now convey the topic on how to find the Nash equilibrium. We already saw cases like the prisoner-dilemma where all players have a dominating strategy that doesn't depend from the other players' actions. This is the "easy" case, as the "confess" is the best-respons strategy for both and hence _(confess,confess)_ is indeed a Nash equilibrium. 

### 2-players game

Sometimes, when the game is small, we can look at the playoff matrix and directly apply the definition of Nash equilibrium to check if candidate strategies are Nash equilibrium.

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
payoff = expand_dimensions(U) 
s      = [[0.5,0.5],[0.5,0.5]]
is_nash(payoff,s) # false
```

Let's veryfying it. To verify that a strategy profile is not a Nash equilibrium it is enought to show that one of the player has a better strategy available conditional to the other player(s) playing the given strategy.

Let's compute the payoff for the two players when they play the given strategy profile and check a different strategy for player 2:

```julia
expected_payoff(payoff,s)                     # (2.5,2.5)
expected_payoff(payoff,[[0.5,0.5],[0.4,0.6]]) # (2.6, 2.6) 
```
From the results above we notice that `(0.5,0.5)` is NOT a Nash equilibrium, player 2 has better respons strategies.
I let to you as an exercise to find a better strategy also for player 1 and verifying instead that the two pure strategies `(1,0)` and `(0,1)` are indeed Nash equilibria.

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

is_nash(payoff,[[1,0],[1,0]]) # false
is_nash(payoff,[[1,0],[0,1]]) # false
is_nash(payoff,[[0,1],[0,1]]) # false
is_nash(payoff,[[0,1],[1,0]]) # false
is_nash(payoff,[[0.5,0.5],[0.5,0.5]]) # true
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

#### 2 players, multiple actions game

Going beyond 2x2 games risks to become quickly intractable.

There are indeed no known algorithm that can _compute_ Nash Equilibrium in polynomial time with the size of the problem (action space), even more to answer questions like if there is a unique equilibrium, if there is a pareto efficient equilibrium, if an equilibrium whose expected payoff for player _n_ is at least _x_.... 
Neverthless, we saw that _verify_ if a given solution (an action profile) is a Nash Equilibrium, is relativelly computationally cheap, and some newest algorithms try to exploit this property.

Here we present two very different approaches. 

##### The Linear Complementarity formulation

The fist one is a Linear Complementarity formulation (LCP), a mathematical programming problem that were origninally solved by Lemke-Howson (1964) using a pivoting procedure.
This mathematical problem can today be solved with other methods (the solver emploied by `StrategicGames` uses the interior point method). Altougth the algorithm has a worst case exponential time with the size of the problem, it remains relativelly fast in practice. The specific equilibrium that is retrieved depends from the initial conditions.

The LCP method finds the equilibrium conditions by exploiting a lot what a "game" is and the characteristics that a (Nash) equilibrium must have. In algeabric terms for a two-players game the problem corresponds to the following linear problem (except the complementarity conditions that are quadratic):

(eq. 1) $~~\sum_{k \in A_2} u_1(a_1^j, a_2^k) * s_2^k + r_1^j = U_1^* ~~~~ \forall j \in A_1$

(eq. 2) $~~\sum_{j \in A_1} u_2(a_1^j, a_2^k) * s_1^j + r_2^k = U_2^* ~~~~ \forall k \in A_2$ 

(eq. 3) $~~\sum_{j \in A_1} s_1^j = 1, ~~ \sum_{k \in A_2} s_2^k = 1$

(eq. 4) $~~s_1^j \geq 0, ~ s_2^k \geq 0 ~~~~~~~~~~~~~~~~~~ \forall j \in A_1, \forall k \in A_2$

(eq. 5) $~~r_1^j \geq 0, ~ r_2^k \geq 0 ~~~~~~~~~~~~~~~~~~ \forall j \in A_1, \forall k \in A_2$

(eq. 6) $~~r_1^j * s_1^j =0, r_2^k * s_2^k =0 ~~~~~ \forall j \in A_1, \forall k \in A_2$

Where $u_1$ and $u_2$ are the $j \times k$ payoff matrices for the two players (a parameter here), while $s_1, s_2$ (the strategies for the two players), $U_1^*, U_2^*$ (the equilibrium expected utility for any action in the support of the two players) and $r_1, r_2$ (the so-called "slack" variables) are the decision variables of the problem (what we want to find).   

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

##### The Support Enumeration Method

The support enumeration method works by replacing the complementarity equation of the LCP formulation by working on subproblems that are linear (at least in two players game) given a certain assumed support of the solution profile at the equilibrium, i.e. the sets of actions $\sigma_1 \subset A_1$ and $\sigma_2 \subset A_2$ that at equilibrium have positive non zero probabilities of being played:

(eq. 1) $~~~~\sum_{k \in \sigma_2} u_1(a_1^j, a_2^k) * s_2^k  = U_1^* ~~~~ \forall j \in \sigma_1$

(eq. 1.2) $~~\sum_{k \in \sigma_2} u_1(a_1^j, a_2^k) * s_2^k  \leq U_1^* ~~~~ \forall j \notin \sigma_1$

(eq. 2) $~~~~\sum_{j \in \sigma_1} u_2(a_1^j, a_2^k) * s_1^j = U_2^* ~~~~ \forall k \in \sigma_2$ 

(eq. 2.2) $~~\sum_{j \in \sigma_1} u_2(a_1^j, a_2^k) * s_1^j \leq U_2^* ~~~~ \forall k \notin \sigma_2$ 

(eq. 3) $~~~~\sum_{j \in A_1} s_1^j = 1, ~~ \sum_{k \in A_2} s_2^k = 1$

(eq. 4) $~~~~s_1^j \geq 0, ~ s_2^k \geq 0 ~~~~~~~~~~~~~~~~~~ \forall j \in \sigma_1, \forall k \in \sigma_2$

(eq. 4.2) $~~s_1^j = 0, ~ s_2^k = 0 ~~~~~~~~~~~~~~~~~~ \forall j \notin \sigma_1, \forall k \notin \sigma_2$

Where the equations have the same meaning as in the LCP formulation.

Of course, the problem is now to find which is the correct support of the equilibrium.

The Porter & oth. (2004) algorithm exploits a smart heuristic to search through all the possible support sets.
It starts by trying small, similar in size supports, to gradually test larger ones and employing a trick named "conditional domination" to exclude certain possible supports from the search.

To implement in Julia the support enumeration method for two players games let's first implement the optimisation problem conditional to a specific support as described in the equations above:

```julia
using HiGHS # HiGHS is a linear solver engine

function nash_on_support_2p(payoff,support= collect.(range.(1,size(payoff)[1:end-1]));solver="HiGHS",verbosity=STD)  
    #println("boo")
    verbosity == FULL && println("Looking for NEq on support: $support")
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function works only for 2 players games")
    (length(nActions) == nPlayers) || error("Mismatch dimension or size between the payoff array and the number of players")
    if isempty(support)
        support = [collect(1:nActions[d]) for d in 1:length(nActions)]
    end
    init=[fill(1/size(payoff,1), size(payoff,1)),fill(1/size(payoff,2), size(payoff,2))]
    m = Model(getfield(eval(Symbol(solver)),:Optimizer))
    if solver == "HiGHS" && verbosity <= STD
        set_optimizer_attribute(m, "output_flag", false)
    end
    @variables m begin
        u[n in 1:2]
    end
    @variable(m, 0-eps() <= s1[j in 1:nActions[1] ] <= 1,  start=init[1][j])
    @variable(m, 0-eps() <= s2[j in 1:nActions[2] ] <= 1,  start=init[2][j])
    for Σ1 in setdiff(1:nActions[1],support[1])
        fix(s1[Σ1], 0.0; force=true);
    end
    for Σ2 in setdiff(1:nActions[2],support[2])
        fix(s2[Σ2], 0.0; force=true);
    end
    @constraints m begin
        utility1_insupport[σ1 in support[1]], # the expected utility for each action in the support of player 1 must be constant
            sum( payoff[σ1,j2,1] * s2[j2] for j2 in 1:nActions[2] )  == u[1]
        utility1_outsupport[Σ1 in setdiff(1:nActions[1],support[1]) ], # the expected utility for each action not in the support of player 1 must be lower than the costant utility above
            sum( payoff[Σ1,j2,1] * s2[j2] for j2 in 1:nActions[2] )  <= u[1]
        utility2_insupport[σ2 in support[2]], # the expected utility for each action in the support of player 2 must be constant
            sum( payoff[j1,σ2,2] * s1[j1] for j1 in 1:nActions[1] )  == u[2]
        utility2_outsupport[Σ2 in setdiff(1:nActions[2],support[2])], # the expected utility for each action not in the support of player 2 must be lower than the costant utility above
            sum( payoff[j1,Σ2,2] * s1[j1] for j1 in 1:nActions[1] )  <= u[2]
        probabilities1,
            sum(s1[j] for j in 1:nActions[1]) == 1
        probabilities2,
            sum(s2[j] for j in 1:nActions[2]) == 1
    end
    @objective m Max u[1] + u[2]
    if verbosity == FULL
        println("Optimisation model to be solved:")
        println(m)
    end
    optimize!(m)
    status = termination_status(m)
    optStrategies = Vector{Vector{Float64}}()
    optU          = Float64[]
    solved = false
    if (status in [MOI.OPTIMAL, MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED, MOI.ALMOST_OPTIMAL]) && has_values(m)
        optStrategies1 = value.(s1)
        optStrategies2 = value.(s2)
        optStrategies = [optStrategies1,optStrategies2]
        optU = value.(u)
        solved = true
    elseif (status in [MOI.LOCALLY_INFEASIBLE, MOI.INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE, MOI.ALMOST_DUAL_INFEASIBLE,MOI.INFEASIBLE_OR_UNBOUNDED, MOI.DUAL_INFEASIBLE])
        solved = false
    else
        if verbosity >= STD
            warn("The feasibility check for support $support returned neither solved neither unsolved ($status). Returning no Nash equilibrium for this support.")
        end
        solved = false
    end
    return (status=status,equilibrium_strategies=optStrategies,expected_payoffs=optU,solved)
end
```

Now let's implement a function that returns the dominated strategies (if any) for a given player:

```julia
function dominated_strategies_2p(payoff,player)
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function supports only 2 players")
    payoff_n = selectdim(payoff,nPlayers+1,player)
    dominated = Int64[]
    if player == 1
        for (ridx,r) in enumerate(eachrow(payoff_n))
            for r2 in eachrow(payoff_n)
                if all(r2 .> r) # r2 strictly dominates r1 
                    push!(dominated,ridx)
                    break
                end
            end
        end
    else
        for (cidx,c) in enumerate(eachcol(payoff_n))
            for c2 in eachcol(payoff_n)
                if all(c2 .> c) # c2 strictly dominates c1 
                    push!(dominated,cidx)
                    break
                end
            end
        end
    end
    return dominated
end
```

Finally we are in the position to implement the support enumeration method. Note that in this version we can ask the algorithm to retrieve one or all the equilibria and if we want to restrict ourselves to only pure Nash equilibria:

```julia
using Combinatorics

"""
    nash_se2(payoff; allow_mixed=true, max_samples=1, verbosity=STD)

Solves Nash eqs using support enumeration for 2 players game using strictly the approach of [Porter-Nudelman-Shoham (2008)](https://doi.org/10.1016/j.geb.2006.03.015)
"""
function nash_se2(payoff; allow_mixed=true, max_samples=Inf, verbosity=STD)
    nActions = size(payoff)[1:end-1]
    nPlayers = size(payoff)[end]
    nPlayers == 2 || error("This function works only for 2 players games")
    nSupportSizes = allow_mixed ? prod(nActions) : 1
    eqs = NamedTuple{(:equilibrium_strategies, :expected_payoffs, :supports), Tuple{Vector{Vector{Float64}}, Vector{Float64},Vector{Vector{Int64}}}}[]
    support_sizes = Matrix{Union{Int64,NTuple{nPlayers,Int64}}}(undef,nSupportSizes,3) # sum, diff, support sizes
    if allow_mixed
        i = 1
        for idx in CartesianIndices(nActions)
            support_sizes[i,:] = [sum(Tuple(idx)),maximum(Tuple(idx))-minimum(Tuple(idx)),Tuple(idx)] 
            i += 1
        end
    else
        support_sizes[1,:] = [nPlayers,0,(ones(Int64,nPlayers)...,)]
    end
    support_sizes = sortslices(support_sizes,dims=1,by=x->(x[2],x[1]))
 
    for support_size in eachrow(support_sizes)
        for S1 in combinations(1:nActions[1],support_size[3][1])
            A2 = setdiff(1:nActions[2],dominated_strategies_2p(payoff[S1,:,:],2))
            if !isempty(dominated_strategies_2p(payoff[S1,A2,:],1))
                continue
            end
            for S2 in combinations(A2,support_size[3][2])
                if !isempty(dominated_strategies_2p(payoff[S1,S2,:],1))
                        continue
                end
                eq_test =  nash_on_support_2p(payoff,[S1,S2],verbosity=verbosity)
                if eq_test.solved
                        push!(eqs,(equilibrium_strategies=eq_test.equilibrium_strategies, expected_payoffs=eq_test.expected_payoffs,supports=[S1,S2]))
                        if length(eqs) == max_samples
                            return eqs
                        end
                end
            end
        end 
    end
    return eqs
end
```

We can try the support enumeration method on the same problems:

```julia
# Head or tail
U = [(1,-1) (-1,1); (-1,1) (1, -1)]
eqs = nash_se2(expand_dimensions(U))
eq_strategies = eqs[1].equilibrium_strategies # [[0.5, 0.5],[0.5,0.5]]

# A biased penalty kick game (kicker - the row player - is more efficient on B)
U = [(-1,1) (1,-1); (1,-1) (0, 0)]
eqs = nash_se2(expand_dimensions(U))
eq_strategies = eqs[1].equilibrium_strategies # [[0.33, 0.66],[0.33,0.66]]

# Battle of the sex (with 3 equilibria)
U = [(2,1) (0,0); (0,0) (1,2)]
eqs = nash_se2(expand_dimensions(U)) 
eq_strategies1 = eqs[1].equilibrium_strategies # [[1,0],[1,0]]
eq_strategies2 = eqs[2].equilibrium_strategies # [[0,1],[0,1]]
eq_strategies3 = eqs[3].equilibrium_strategies # [[0.66,0.33],[0.33,0.66]]

# A 2-players game with 2 and 3 actions respectively
U = [(1,-1) (-1,1) (1,0); (-1,1) (1, -1) (0,1)]
eqs = nash_se2(expand_dimensions(U)) 
eq_strategies = eqs[1].equilibrium_strategies # [[0.66, 0.33],[0, 0.33, 0.66]]
```

### Multiple players game

The algorithms in the section above can be generalised to work with generic N players, altought in most cases the optimisation problem will be no longer linear. 
All functions in the `StrategicGames` package have been extended to work with N players, where each player can have a different actions space. In particular the `nash_cp(payoff;init)` and `nash_se(payoff;allow_mixed, max_samples)` functions retrieve a/the Nash equilibria for all standard form games (subject to enought computational resources).

For example:

```julia
# This example is taken from https://www.youtube.com/watch?v=bKrwQKUT0v8 where it is analytically solved
U = [(0,0,0) ; (3,3,3) ;; (3,3,3) ; (2,2,4) ;;;
     (3,3,3) ; (2,4,2) ;; (4,2,2) ; (1,1,1) ;;;]
eq = nash_cp(expand_dimensions(U))
eq_strategies = eq.equilibrium_strategies
p = -1 + sqrt(10)/2 # approximatively 0.58
eq_strategies ≈ [[p,1-p],[p,1-p],[p,1-p]] # true

eqs = nash_se(expand_dimensions(U),max_samples=Inf)
n_eqs = length(eqs) # 7
eq_strategies1 = eqs[1].equilibrium_strategies # [[0,1],[1,0],[1,0]]
eq_strategies2 = eqs[2].equilibrium_strategies # [[1,0],[0,1],[1,0]]
eq_strategies3 = eqs[3].equilibrium_strategies # [[1,0],[1,0],[0,1]]
eq_strategies4 = eqs[4].equilibrium_strategies # [[0.25,0.75],[0.25,0.75],[1,0]]
eq_strategies5 = eqs[5].equilibrium_strategies # [[0.25,0.75],[1,0],[0.25,0.75]]
eq_strategies6 = eqs[6].equilibrium_strategies # [[1,0],[0.25,0.75],[0.25,0.75]]
eq_strategies7 = eqs[7].equilibrium_strategies # [[0.58, 0.42][0.58, 0.42],[0.58, 0.42]]
```