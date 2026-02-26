# Adaptive & Emergent Taxonomy Systems — Research Findings

**Date:** 2026-02-26
**Status:** Research complete
**Context:** Informing taxonomy design for lessons-db knowledge capture extension
**Method:** Literature review + system analysis (12 sources, 5 topic areas)

---

## BLUF

Fixed categories applied too early calcify wrong assumptions and cause missed patterns (Wason: 90% failure rate on hypothesis testing). The strongest systems start with free-form tagging, use vector embeddings to discover natural clusters, then selectively solidify clusters into categories only after sufficient data accumulates. The literature consistently recommends a **three-phase pattern**: capture freely, cluster computationally, formalize selectively.

---

## 1. Dynamic Clustering vs Static Categories

### What the Literature Says

**Confidence: High** — Multiple independent sources converge on the same conclusion.

**Static categories fail when the domain is evolving or poorly understood.** Shirky (2005) argues that pre-designed ontologies are "a 300-year-old hack" that forces organizers to "have responsibility to organize the world in advance, necessarily overriding user needs." The Library of Congress system exists not because concepts need hierarchies — but because physical books need shelf locations. Digital knowledge has no such constraint.

**Premature taxonomy embeds historical bias permanently.** Shirky's examples:
- Dewey Decimal overrepresents Christian theology (200s: 8 of 10 slots) and treats entire continents as equivalent to individual European nations
- Soviet libraries placed "Marxism-Leninism" as the root category
- The "noble gases" name persists despite being based on a non-essential property (gaseous at room temperature)

**Dynamic clustering outperforms static when:** the domain is evolving, user mental models vary significantly, or the corpus is growing faster than expert curation can handle. Tucker (2020) found that effective taxonomies "adapt and change in coevolution with the efforts of users to make sense of ambiguity, emergence, and uncertainty."

**Static categories are appropriate when:** the domain is well-understood, the vocabulary is stable, regulatory compliance requires fixed terms, or physical objects need shelf locations.

### Practical Decision Framework

| Signal | Use Static | Use Dynamic |
|--------|-----------|-------------|
| Domain maturity | Mature, stable vocabulary | Evolving, new concepts emerging |
| Corpus size | Small, manually curated | Large, growing autonomously |
| User mental models | Uniform across users | Diverse, context-dependent |
| Compliance needs | Regulatory requirements | Internal knowledge only |
| Update frequency | Rarely reclassified | Frequently recategorized |

### Implication for lessons-db

The current 6 clusters (A-F) and 10 categories were derived from 122 negative lessons in a specific domain. Extending to positive knowledge (patterns, innovations, value multipliers) means entering a poorly-understood domain where premature categorization will likely embed the wrong structure. **Start dynamic, solidify later.**

**Sources:**
- [Shirky: Ontology is Overrated (2005)](https://gwern.net/doc/philosophy/ontology/2005-04-shirky-ontologyisoverratedcategorieslinksandtags.html)
- [Tucker: Taxonomy design methodologies (2020)](https://journals.sagepub.com/doi/abs/10.1177/0340035219877206)
- [Enterprise Knowledge: Enhancing Taxonomy Management Through Knowledge Intelligence](https://enterprise-knowledge.com/enhancing-taxonomy-management-through-knowledge-intelligence/)

---

## 2. Emergent Tagging Systems — Real-World Implementations

### Stack Overflow: Folksonomy with Guardrails

**Confidence: High** — First-party blog post documenting their actual evolution.

Stack Overflow's tag system is the best-documented case of emergent taxonomy at scale. Key evolution:

1. **Phase 1 — Open tagging.** Anyone could create tags. Result: `[js]`, `[javascript]`, `[java-script]` all existed simultaneously. Content fragmented across synonyms.

2. **Phase 2 — Friction gates.** Progressively raised reputation threshold for tag creation: 250 → 500 → 1,500 rep. Reduced noise without eliminating emergence.

3. **Phase 3 — Synonym system.** Community-driven synonym proposals. `[js]` → automatically remapped to `[javascript]`. Preserved the folksonomy but unified retrieval.

4. **Phase 4 — Automated cleanup.** Single-use tags older than 6 months auto-culled. Dead-end experiments cleaned without manual moderation.

**Critical insight:** "Rather than predicting all possible variants, they observed what users actually created and based synonyms on real usage patterns, not theoretical predictions."

### Wikipedia: Category Graph Problems

**Confidence: High** — Academic studies on real data.

Wikipedia's category system (~400,000 categories) demonstrates the failure mode of unrestricted emergent categorization without governance:
- The category graph is directed but NOT acyclic — cycles create navigation confusion
- Naive subcategory inference causes type errors (e.g., every "Electronic Rock Song" getting typed as "Genre" through category inheritance)
- Articles in coarse-grained categories are less likely to reach featured quality — broader categories correlate with worse content (PMC 5749832)

**Lesson:** Emergence without any governance produces a mess. The winning pattern is **governed emergence** — free creation with periodic consolidation.

### Thomas Vander Wal's Folksonomy Framework

**Confidence: High** — Coined the term, well-documented framework.

Vander Wal (2004) distinguishes two folksonomy types:
- **Broad folksonomy:** Multiple users tag the same item → popular tags surface through frequency. Delicious, Stack Overflow.
- **Narrow folksonomy:** Few users (often the creator) tag items → tags reflect individual mental models. Personal wikis, Obsidian vaults.

For a single-user system like lessons-db, narrow folksonomy dynamics apply. Tags reflect one person's mental model, which means:
- Less synonym collision (one person is more consistent than a crowd)
- Higher risk of blind spots (no crowd to surface alternative framings)
- Embedding-based discovery becomes MORE important (compensates for single-viewpoint bias)

**Sources:**
- [Stack Overflow: Tag Folksonomy and Tag Synonyms (2010)](https://stackoverflow.blog/2010/08/01/tag-folksonomy-and-tag-synonyms/)
- [Wikipedia: Folksonomy](https://en.wikipedia.org/wiki/Folksonomy)
- [Vander Wal: Folksonomy](https://vanderwal.net/folksonomy.html)
- [Knowledge categorization affects popularity and quality of Wikipedia articles (PMC 5749832)](https://pmc.ncbi.nlm.nih.gov/articles/PMC5749832/)

---

## 3. Embedding-Based Clustering

### BERTopic Pipeline

**Confidence: High** — Widely adopted, well-documented, reproducible.

BERTopic (Grootendorst, 2022) is the current standard for discovering topics from text embeddings. Pipeline:

1. **Embed** — Sentence-transformers convert documents to dense vectors (default: `all-MiniLM-L6-v2`, 384 dims)
2. **Reduce** — UMAP projects high-dimensional vectors to ~5 dimensions while preserving local structure
3. **Cluster** — HDBSCAN finds density-based clusters without requiring a predefined cluster count. Documents in sparse regions become outliers (topic -1), not forced into bad clusters.
4. **Label** — Class-based TF-IDF (c-TF-IDF) extracts representative terms per cluster

Key properties relevant to lessons-db:
- **No predefined cluster count** — HDBSCAN discovers natural groupings. Only requires `min_cluster_size` (minimum documents to form a cluster).
- **Handles noise** — Outlier detection means items that don't fit any cluster stay unassigned rather than being forced into wrong categories.
- **Modular** — Each pipeline stage is swappable. Can use different embedders, reducers, or clusterers.
- **Dynamic topics** — BERTopic supports tracking topic evolution over time, which maps to watching how lesson categories shift as more knowledge accumulates.

### MongoDB Semantic Vector Clustering

**Confidence: Medium** — Concept well-described, implementation details sparse.

MongoDB's approach to vector clustering emphasizes **self-organizing hierarchies**: "The data itself figures out these groups and self-organizes." Vectors are auto-aggregated into semantic clusters, then those clusters can be summarized into higher-level theme vectors for RAG retrieval.

The key insight: you can cluster at multiple levels of granularity. A lesson might belong to a fine-grained cluster ("WebSocket connection lifecycle") and a coarse cluster ("Integration Boundaries") simultaneously.

### Pinecone's Approach

Pinecone focuses on similarity search rather than clustering per se. Their value for this use case is **query-time grouping** — given a new lesson, find the 10 most similar existing lessons and let the user see what they cluster near, rather than pre-assigning to a cluster.

### LanceDB (Current lessons-db Vector Store)

LanceDB stores vectors locally (no server). It supports ANN search but doesn't have built-in clustering. However, the embeddings it stores can be extracted and clustered externally using HDBSCAN or BERTopic, then cluster labels written back as metadata.

### Practical Implementation Pattern

```
1. Store lessons with free-form tags + embeddings (LanceDB)
2. Periodically run HDBSCAN on all embeddings
3. Compare discovered clusters against existing categories
4. Surface new clusters to the user: "These 8 lessons cluster together
   but don't share a category. Suggested label: [c-TF-IDF terms]"
5. User confirms/rejects/renames → category promoted
6. Repeat as corpus grows
```

This is the **cluster-then-label** pattern — the opposite of traditional **label-then-sort**.

**Sources:**
- [BERTopic: Algorithm](https://maartengr.github.io/BERTopic/algorithm/algorithm.html)
- [BERTopic: Best Practices](https://maartengr.github.io/BERTopic/getting_started/best_practices/best_practices.html)
- [MongoDB: Find Hidden Insights in Vector Databases: Semantic Clustering](https://www.mongodb.com/blog/post/find-hidden-insights-vector-databases-semantic-clustering)
- [A Comprehensive Survey on Deep Clustering (ACM Computing Surveys, 2024)](https://dl.acm.org/doi/10.1145/3689036)

---

## 4. Confirmation Bias in Categorization

### The Wason 2-4-6 Problem

**Confidence: High** — Foundational cognitive science, replicated extensively.

Wason (1960) demonstrated that when given the sequence 2-4-6 and asked to discover the rule, **only 10% of participants succeeded**. The rule was simply "increasing numbers," but participants formed the hypothesis "even numbers increasing by 2" and then only tested confirming examples (4-8-10, 6-8-12, 20-22-24). They never tested disconfirming examples (1-3-5 would have revealed their hypothesis was too narrow).

**This is the "looking for redheads, missing blondes" problem applied to knowledge management.** When categories exist, people:

1. **Seek confirming instances** — A lesson that looks like "Integration Boundary" gets filed there. A lesson that's actually about integration boundaries BUT manifests as a cold-start problem gets filed under "Cold-Start" because the surface symptom matched first.

2. **Stop investigating when a category fits** — Once "this is a Cluster B issue" registers, the search for alternative explanations stops. The lesson might actually represent a new pattern that crosscuts existing categories.

3. **Don't test negative cases** — Nobody asks "what lessons does this cluster NOT contain that it should?" The absence of expected items in a category is invisible.

### The Positive Test Strategy

**Confidence: High** — Confirmed across multiple studies.

Klayman & Ha (1987) formalized this as the "positive test strategy" — people test hypotheses by looking for cases where the predicted outcome occurs, not cases where it shouldn't. In categorization terms:

- We verify that items IN a category belong there (positive test)
- We rarely check if items OUTSIDE a category should be there (negative test)
- We almost never check if the category boundaries themselves are wrong

### Practical Mitigations

From the literature, three approaches reduce confirmation bias in categorization:

1. **Forced alternative generation** — Before assigning a category, require generating at least one alternative category. "This looks like Cluster A, but could it also be Cluster D?" Thinking in opposites improved Wason task performance significantly (PMC 12402026, 2025).

2. **Embedding-based neighbors** — Show the 5 nearest neighbors in vector space regardless of assigned category. If nearest neighbors span multiple categories, the current category assignment may be wrong.

3. **Periodic cluster recomputation** — Don't just add new items to existing clusters. Periodically recluster everything from scratch. New items may retroactively reveal that old items were miscategorized.

**Sources:**
- [Wason: On the failure to eliminate hypotheses in a conceptual task (1960)](https://explorable.com/confirmation-bias)
- [Nickerson: Confirmation Bias: A Ubiquitous Phenomenon in Many Guises (1998)](https://pages.ucsd.edu/~mckenzie/nickersonConfirmationBias.pdf)
- [Thinking in opposites improves hypothesis testing (PMC 12402026, 2025)](https://pmc.ncbi.nlm.nih.gov/articles/PMC12402026/)
- [Confirmation bias in perceptual decision-making due to hierarchical approximate inference (PMC 8659691)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8659691/)

---

## 5. Adaptive Taxonomy Patterns

### Pattern: Progressive Formalization

**Confidence: High** — Converges across PKM, enterprise KM, and information science.

The strongest pattern across all sources is what I'm calling **progressive formalization** — a three-phase lifecycle:

**Phase 1: Free Capture** (days → weeks)
- Tags are free-form strings, no controlled vocabulary
- Every item gets an embedding vector
- No hierarchy, no categories, no structure imposed
- Equivalent to: Zettelkasten daily notes, Roam Research block-level tagging

**Phase 2: Computational Discovery** (weekly → monthly)
- Cluster embeddings to find natural groupings
- Surface cluster labels (c-TF-IDF or LLM-generated)
- Show tag co-occurrence patterns
- Propose merges for synonymous tags
- Equivalent to: BERTopic dynamic topic modeling, Stack Overflow synonym detection

**Phase 3: Selective Solidification** (monthly → quarterly)
- Promote stable clusters to named categories
- Categories that persist across 3+ clustering runs are "real"
- Categories that fragment or merge are still emergent
- Governance: user confirms promotions, system proposes
- Equivalent to: Stack Overflow's graduated tag privileges, Enterprise Knowledge's KI framework

### PKM Tools: How They Handle Emergence

**Nick Milo's Maps of Content (MOC):**
- Notes accumulate freely. When a critical mass of related notes exists (~20-30), the user creates a MOC — a note that links to all related notes and describes their relationships.
- MOCs are "work stations, places to examine how notes interact" — not just indexes but active thinking tools.
- Structure manifests "bottom-up" — no folders, only linking hubs that emerge from usage.
- **Key principle:** "Rather than being a representation of previously made connections, MOCs are where connections are made anew."

**Tiago Forte's PARA + Progressive Summarization:**
- PARA provides 4 fixed top-level categories (Projects, Areas, Resources, Archives) as a minimal scaffold.
- Within each, organization is deferred via progressive summarization — information is highlighted in layers over time, with structure emerging from repeated revisitation.
- **Key principle:** "Summarizing and condensing a piece of information in small spurts, spread across time, in the course of other work."

**Roam Research / LogSeq:**
- Block-level references create emergent structure through usage.
- Child blocks inherit parent tags, enabling queries that pull together content without explicit categorization.
- Structure "can be more emergent and less pre-defined" than page-based systems.

### Enterprise KM: The Knowledge Intelligence Framework

**Confidence: Medium** — Industry framework, limited independent validation.

Enterprise Knowledge's KI framework operationalizes progressive formalization:

1. **Seed** — Domain experts define core concepts (small, intentionally incomplete)
2. **Enrich** — NLP/BERTopic analyze organizational content, extract candidate terms and relationships
3. **Validate** — Every suggested change goes through governance: automated compliance check → human review → documentation → version control
4. **Refine** — User interaction data (search patterns, click-through) feeds back to improve the taxonomy

**Critical design element:** "Every suggested change, whether generated through user behavior or content analysis, goes through a structured governance process." The system never auto-modifies the taxonomy — it proposes, humans approve.

**Sources:**
- [Nick Milo: In what ways can we form useful relationships between notes?](https://medium.com/@nickmilo22/in-what-ways-can-we-form-useful-relationships-between-notes-9b9ec46973c6)
- [Forte: Progressive Summarization](https://fortelabs.com/blog/progressive-summarization-a-practical-technique-for-designing-discoverable-notes/)
- [Bob Doto: Zettelkasten, Linking Your Thinking, and Nick Milo's Search for Ground](https://writing.bobdoto.computer/zettelkasten-linking-your-thinking-and-nick-milos-search-for-ground/)
- [Enterprise Knowledge: Enhancing Taxonomy Management](https://enterprise-knowledge.com/enhancing-taxonomy-management-through-knowledge-intelligence/)

---

## 6. Synthesis — Design Implications

### Findings → Implications → Recommendations (RAND separation)

**Finding 1:** Premature categorization embeds bias and misses patterns (Shirky, Wason).
**Implication:** The current 6 clusters and 10 categories should NOT be extended to cover positive knowledge types.
**Recommendation:** New knowledge types enter with free-form tags only. Existing negative-lesson categories remain (they're battle-tested with 122 items). New categories for positive knowledge emerge from data.

**Finding 2:** Single-user folksonomies have less synonym noise but higher blind-spot risk (Vander Wal).
**Implication:** Justin's tagging will be consistent but potentially narrow. No crowd to surface alternative framings.
**Recommendation:** Use embedding neighbors as a "synthetic crowd" — when a lesson is captured, show the 5 nearest neighbors regardless of category to surface unexpected connections.

**Finding 3:** HDBSCAN discovers natural clusters without requiring predefined count (BERTopic).
**Implication:** Cluster count can grow organically as the corpus grows. No need to decide "how many categories" upfront.
**Recommendation:** Periodic (weekly or on-demand) reclustering of all embeddings. Surface new/changed clusters as suggestions, not automatic reclassifications.

**Finding 4:** Stack Overflow's graduated approach (open → friction gates → synonyms → auto-cleanup) worked at scale.
**Implication:** The progression from "anyone can tag" to "tags require evidence" can be modeled as tag maturity.
**Recommendation:** Track tag usage count and age. Tags used 1x after 6 months get flagged for review (Stack Overflow's cleanup rule). Tags used 5+ times across 3+ months get promoted to "established."

**Finding 5:** MOCs and progressive summarization defer structure until it's earned.
**Implication:** Category assignment at capture time is premature for new knowledge types.
**Recommendation:** Capture with tags + embedding. Category assignment happens at review time (weekly/monthly), not capture time. The system suggests categories; the user confirms.

### Confidence Assessment

| Recommendation | Confidence | Evidence Strength |
|---------------|------------|-------------------|
| Free-form tags for new knowledge types | High | Shirky + Vander Wal + Stack Overflow + PKM tools all converge |
| Embedding-based cluster discovery | High | BERTopic well-validated, MongoDB/LanceDB implementations exist |
| Periodic reclustering | Medium | Theoretically sound, limited empirical data on frequency |
| Tag maturity lifecycle | Medium | Stack Overflow model validated at scale, untested for single-user |
| Embedding neighbors as bias mitigation | Medium | Cognitively sound (forced alternative generation), no direct study |
| Separate new categories from existing | High | Premature taxonomy literature strongly supports |

### Open Questions

1. **Cluster stability threshold** — How many items and how many clustering runs before a cluster should be promoted to a named category? (Needs empirical testing)
2. **Embedding model choice** — Current lessons-db uses `nomic-embed-text` (768 dims) via Ollama. Is this sufficient for cross-domain knowledge clustering, or does the domain shift from "coding mistakes" to "positive patterns" require a different model?
3. **Reclustering frequency** — Weekly? On every 10 new items? On-demand only? Trade-off between freshness and stability.
4. **Cross-cutting membership** — Should items belong to exactly one cluster (hard assignment) or multiple clusters (soft assignment / fuzzy membership)? HDBSCAN supports soft clustering via `prediction_data=True`.

---

## Appendix: Key Terminology

| Term | Definition | Source |
|------|-----------|--------|
| **Folksonomy** | User-generated classification through free tagging | Vander Wal 2004 |
| **Broad folksonomy** | Multiple users tag same items; popular tags emerge via frequency | Vander Wal 2004 |
| **Narrow folksonomy** | Few users (often creator) tag items; reflects individual mental model | Vander Wal 2004 |
| **Progressive formalization** | Structure emerges over time from usage, not upfront design | Synthesized from PKM + KI literature |
| **c-TF-IDF** | Class-based TF-IDF — extracts representative terms per cluster | Grootendorst 2022 (BERTopic) |
| **HDBSCAN** | Hierarchical Density-Based Spatial Clustering; finds natural clusters without predefined count | Campello et al. 2013 |
| **MOC** | Map of Content — emergent linking hub for related notes | Nick Milo (LYT) |
| **Positive test strategy** | Testing hypotheses by seeking confirming evidence only | Klayman & Ha 1987 |
| **KI Framework** | Knowledge Intelligence — AI-augmented taxonomy management cycle | Enterprise Knowledge |
