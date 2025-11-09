def generate_double_round_robin(teams):
    n = len(teams)
    assert n % 2 == 0, "Teams must be even"

    T = list(teams)
    rounds = []

    for r in range(n - 1):
        pairings = []
        for i in range(n // 2):
            t1, t2 = T[i], T[n - 1 - i]
            pairings.append((t1, t2) if r % 2 == 0 else (t2, t1))
        rounds.append(pairings)
        T = [T[0]] + [T[-1]] + T[1:-1]

    second_half = [[(b, a) for (a, b) in rnd] for rnd in rounds]
    return rounds + second_half
