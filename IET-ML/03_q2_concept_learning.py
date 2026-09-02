"""
Concept Learning formulation:
Target concept c: "HazardousDay(x)" -> {Yes, No}
We discretize continuous attributes into symbolic values so classic
Candidate-Elimination (Mitchell) can be applied.

Attributes (discretized):
  PM2.5    : {Low, Medium, High}
  NO2      : {Low, High}
  WindSpeed: {Calm, Windy}
  Humidity : {Low, High}

Hypothesis representation: conjunction of constraints, each attribute value
can be a specific value, '?' (any value accepted) or a null/'0' hypothesis
(no value accepted) -- the classic Find-S / Candidate-Elimination hypothesis
language.
"""
import pandas as pd

def discretize(row):
    pm25 = "Low" if row["PM2.5"] < 90 else ("Medium" if row["PM2.5"] < 130 else "High")
    no2 = "Low" if row["NO2"] < 45 else "High"
    wind = "Calm" if row["WindSpeed"] < 3 else "Windy"
    hum = "Low" if row["Humidity"] < 55 else "High"
    return pd.Series([pm25, no2, wind, hum])

df = pd.read_csv("data/air_quality_cleaned.csv")
disc = df.apply(discretize, axis=1)
disc.columns = ["PM2.5","NO2","WindSpeed","Humidity"]
disc["Hazardous"] = (df["AQI_Category"] == "Hazardous").map({True:"Yes", False:"No"})

# Build a small, clean training set of 8 representative examples for manual CE
uniq = disc.drop_duplicates(subset=["PM2.5","NO2","WindSpeed","Humidity"])
pos = uniq[uniq["Hazardous"] == "Yes"].head(4)
neg = uniq[uniq["Hazardous"] == "No"].head(4)
examples = pd.concat([pos, neg]).reset_index(drop=True)
# interleave: pos, neg, pos, neg... so CE trace is illustrative
examples = pd.concat([examples.iloc[0::2].reset_index(drop=True),
                       examples.iloc[1::2].reset_index(drop=True)]).reset_index(drop=True)
order_idx = [0,4,1,5,2,6,3,7]
examples = examples.iloc[[i for i in order_idx if i < len(examples)]].reset_index(drop=True)
print("Training examples used for Candidate Elimination:")
print(examples.to_string())
examples.to_csv("data/concept_learning_examples.csv", index=False)

attrs = ["PM2.5","NO2","WindSpeed","Humidity"]

def more_general(h1, h2):
    """Return True if h1 is more-or-equally general than h2."""
    for a,b in zip(h1,h2):
        if a == '?': continue
        if a == '0': return False
        if a != b: return False
    return True

def consistent(h, example):
    for a, v in zip(h, example):
        if a == '0': return False
        if a != '?' and a != v: return False
    return True

def generalize_S(s, example):
    new_s = list(s)
    for i,(a,v) in enumerate(zip(s, example)):
        if a == '0':
            new_s[i] = v
        elif a != v:
            new_s[i] = '?'
    return tuple(new_s)

def specialize_G(g, example, attr_domains, S):
    specializations = []
    for i, a in enumerate(g):
        if a == '?':
            for val in attr_domains[i]:
                if val != example[i]:
                    new_g = list(g); new_g[i] = val
                    specializations.append(tuple(new_g))
    # keep only those consistent with S (more general than S)
    return [sp for sp in specializations if more_general(sp, S)]

attr_domains = [sorted(disc[a].unique().tolist()) for a in attrs]

S = tuple(['0']*len(attrs))
G = {tuple(['?']*len(attrs))}

steps = []
for idx, row in examples.iterrows():
    x = tuple(row[a] for a in attrs)
    label = row["Hazardous"]
    if label == "Yes":  # positive example
        # remove G hypotheses inconsistent with x
        G = {g for g in G if consistent(g, x)}
        S = generalize_S(S, x)
        # remove from G any hyp more specific than S is handled implicitly (single S version)
    else:  # negative example
        if consistent(S, x):
            # S wrongly covers a negative -> problem in single-hypothesis simplification (skip, log)
            pass
        newG = set()
        for g in G:
            if consistent(g, x):
                # specialize
                specs = specialize_G(g, x, attr_domains, S)
                newG.update(specs)
            else:
                newG.add(g)
        # keep only most general (remove hypotheses more specific than another in G)
        G = {g for g in newG if not any(g != g2 and more_general(g2, g) for g2 in newG)}
    steps.append((idx, x, label, S, frozenset(G)))

print("\n=== Candidate Elimination trace ===")
for idx, x, label, s, g in steps:
    print(f"Example {idx}: {x} -> {label}")
    print(f"   S = {s}")
    print(f"   G = {sorted(g)}")

print("\nFinal Specific Boundary S:", S)
print("Final General Boundary G:", sorted(G))

with open("data/q2_ce_trace.txt","w") as f:
    f.write("Attributes order: " + str(attrs) + "\n\n")
    for idx, x, label, s, g in steps:
        f.write(f"Example {idx}: {x} -> {label}\n   S={s}\n   G={sorted(g)}\n\n")
    f.write(f"Final S = {S}\nFinal G = {sorted(G)}\n")
