from __future__ import annotations

CONCEPT_TERMS: dict[str, list[str]] = {
    "biology": [
        "biology", "biological", "cell", "cells", "organism", "gene", "genetic", "dna", "rna",
        "protein", "enzyme", "evolution", "species", "photosynthesis", "respiration",
        "chromosome", "mitosis", "meiosis", "bacteria", "virus", "tissue", "metabolism",
        "ecosystem", "neuron", "hormone", "immune",
    ],
    "physics": [
        "physics", "physical", "force", "energy", "momentum", "velocity", "acceleration",
        "mass", "gravity", "electric", "magnetic", "electromagnetic", "quantum", "relativity",
        "thermodynamics", "entropy", "wave", "particle", "voltage", "current", "charge",
        "friction", "kinetic", "potential energy", "photon", "atom",
    ],
    "chemistry": [
        "chemistry", "chemical", "molecule", "molecular", "atom", "atomic", "element", "compound",
        "reaction", "bond", "covalent", "ionic", "acid", "base", "solution", "solvent", "mole",
        "stoichiometry", "oxidation", "reduction", "catalyst", "equilibrium", "titration",
        "periodic table", "isotope", "electron",
    ],
    "mathematics": [
        "mathematics", "mathematical", "algebra", "algebraic", "equation", "function",
        "derivative", "integral", "calculus", "geometry", "geometric", "theorem", "proof",
        "polynomial", "matrix", "vector", "logarithm", "trigonometry", "sine", "cosine",
        "arithmetic", "integer", "fraction", "exponent", "coefficient",
    ],
    "statistics": [
        "statistics", "statistical", "probability", "distribution", "mean", "median", "variance",
        "standard deviation", "correlation", "regression", "hypothesis", "p-value", "significance",
        "confidence interval", "sample", "sampling", "population", "normal distribution",
        "null hypothesis", "t-test", "chi-square", "random variable", "expected value", "percentile",
    ],
    "computer_science": [
        "computer science", "computer", "algorithm", "algorithmic", "program", "programming",
        "code", "software", "data structure", "array", "stack", "queue", "hash", "recursion",
        "loop", "compiler", "binary", "bit", "byte", "memory", "database", "complexity",
        "sorting", "boolean", "pseudocode",
    ],


    "wmdp-bio": [
        "virus", "viral", "virulence", "pathogen", "pathogenic", "bacterium", "bacterial",
        "toxin", "infectious", "infectivity", "transmissibility", "host range", "tropism",
        "gene editing", "reverse genetics", "plasmid", "vector", "viral vector", "strain",
        "genome", "genomic", "mutation", "serotype", "spore", "culture", "assay", "vaccine",
        "antibody", "immune", "select agent", "bioweapon", "biosecurity", "pandemic",
        "gain of function", "aerosol", "pathogenesis", "replication",
    ],
    "wmdp-cyber": [
        "exploit", "exploitation", "vulnerability", "payload", "shellcode", "malware",
        "ransomware", "buffer overflow", "privilege escalation", "reverse shell", "backdoor",
        "rootkit", "injection", "cve", "reconnaissance", "enumeration", "port scan", "nmap",
        "metasploit", "obfuscation", "persistence", "command and control", "exfiltration",
        "authentication", "encryption", "hash", "packet", "protocol", "firewall", "kernel",
        "memory corruption", "race condition", "sandbox escape", "phishing",
    ],
}


def terms_for(concept_id: str) -> list[str]:
    return CONCEPT_TERMS.get(concept_id, [concept_id])


def has_vocabulary(concept_id: str) -> bool:


    return concept_id in CONCEPT_TERMS
