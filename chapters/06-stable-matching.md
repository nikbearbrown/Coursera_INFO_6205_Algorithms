# Chapter 6 — Stable Matching

## Introduction to Stable Matching

Stable matching is a fundamental concept in algorithm design and game
theory. It involves pairing elements from two sets, such as students and
schools, in a way that no pair of elements would prefer each other over
their current matches. This concept ensures that the matching is stable,
meaning there are no two elements that would rather be paired with each
other than with their current partners.

Stable matching algorithms are crucial in various real-world
applications, from college admissions to job placements. This section
introduces the concept of stable matching, providing a detailed
explanation of its principles and significance. We will explore the
technical definition of stable matching and discuss important related
concepts.

### Concept of Stable Matching

Stable matching addresses the problem of pairing elements from two sets
such that the resulting matches are stable. A matching is considered
stable if there are no two elements, each from different sets, who would
prefer to be matched with each other rather than with their current
partners.

Technically, let $`A = \{a_1, a_2, \ldots, a_n\}`$ and
$`B = \{b_1, b_2, \ldots, b_n\}`$ be two sets, each containing $`n`$
elements. Each element $`a_i \in A`$ has a preference list ranking all
elements in $`B`$, and similarly, each element $`b_j \in B`$ has a
preference list ranking all elements in $`A`$. A matching $`M`$ is a set
of pairs $`(a_i, b_j)`$ such that each element is matched with exactly
one partner from the other set.

A matching $`M`$ is stable if there are no pairs $`(a_i, b_j)`$ and
$`(a_k, b_l) \in M`$ such that $`a_i`$ prefers $`b_l`$ over $`b_j`$ and
$`b_l`$ prefers $`a_i`$ over $`a_k`$. In other words, there are no two
elements that would rather be with each other than with their assigned
partners.

### Real-World Applications

Stable matching algorithms have a wide range of applications in various
fields, where they help optimize and stabilize pairing processes.

**College Admissions:** In the college admissions process, students
apply to colleges, and colleges rank the students. The goal is to match
students to colleges such that no student and college would prefer each
other over their current assignments. The Gale-Shapley algorithm can be
used to create a stable matching between students and colleges, ensuring
that the process is fair and efficient.

**Job Placements:** In job markets, employers and job seekers both have
preferences for their potential matches. Stable matching algorithms help
create optimal pairings that minimize the likelihood of employees
wanting to switch jobs and employers wanting to change their hires.

**Organ Transplants:** In medical fields, particularly in organ
transplant scenarios, stable matching algorithms ensure that organ
donations are matched with recipients in a way that maximizes
compatibility and minimizes waiting times, potentially saving lives.

## Gale-Shapley Algorithm

The Gale-Shapley algorithm, also known as the Deferred Acceptance
algorithm, is a seminal solution to the stable matching problem. This
section provides a comprehensive overview of the algorithm, detailing
its procedure and highlighting each phase of the algorithm.

### Overview

The Gale-Shapley algorithm is designed to find a stable matching between
two equally sized sets of elements, ensuring that no pair of elements
would prefer each other over their current matches. This algorithm
operates by having one set of elements propose to the other set based on
their preferences until a stable matching is achieved.

### Procedure of the Gale-Shapley Algorithm

The procedure of the Gale-Shapley algorithm can be broken down into
three main phases: Initialization, Proposing and Rejecting Phase, and
Termination and Output.

#### Initialization

In the initialization phase, all elements in both sets are marked as
free, indicating that they are not currently matched with any partner.
Additionally, each element in the proposing set (e.g., students)
prepares to propose to elements in the receiving set (e.g., schools)
based on their preference lists.

#### Proposing and Rejecting Phase

During the proposing and rejecting phase, each free element in the
proposing set proposes to the next element on its preference list. If
the proposed element is free, the two become engaged. If the proposed
element is already engaged, it compares the new proposal with its
current partner. If the new proposal is preferred, the current
engagement is broken, and the new pair becomes engaged. Otherwise, the
proposer moves on to the next preference.

<div class="algorithm">

<div class="algorithmic">

$`a_i`$ proposes to the next $`b_j`$ on his preference list
$`(a_i, b_j)`$ become engaged $`b_j`$ breaks the engagement with $`a_k`$
$`(a_i, b_j)`$ become engaged $`a_k`$ becomes free

</div>

</div>

#### Termination and Output

The algorithm terminates when there are no more free elements in the
proposing set that can make further proposals. At this point, the
matching is stable, and the algorithm outputs the set of engaged pairs,
which represents the stable matching.

<div class="algorithm">

<div class="algorithmic">

Output the set of engaged pairs $`(a_i, b_j)`$

</div>

</div>

The Gale-Shapley algorithm guarantees that the matching is stable and
that no pair of elements would prefer each other over their current
matches. This makes it a powerful and widely used solution in various
practical applications, from college admissions to job placements.

## Optimality of the Gale-Shapley Algorithm

The Gale-Shapley algorithm is celebrated not only for its ability to
produce stable matchings but also for its optimality properties. This
section will explore these properties in detail, illustrating the
algorithm’s efficiency and fairness through theoretical insights and
practical examples.

### Optimality Properties

The Gale-Shapley algorithm guarantees two primary forms of optimality:

- **Proposer Optimality:** The algorithm ensures that each proposer
  (e.g., students in the student-proposing version) gets the best
  possible partner according to their preferences. No proposer can
  achieve a better match in any stable matching.

- **Receiver Pessimality:** Conversely, the receivers (e.g., schools)
  get their worst possible partners among all stable matchings. However,
  they still get a stable match, ensuring that no unmatched pair would
  prefer each other over their current matches.

Mathematically, let $`M`$ be the matching produced by the Gale-Shapley
algorithm with proposers $`A`$ and receivers $`B`$. The proposer
optimality can be represented as:
``` math
\forall a_i \in A, \, M(a_i) \geq M'(a_i)
```
for any other stable matching $`M'`$.

### Example Demonstrating Optimality

To illustrate the optimality properties of the Gale-Shapley algorithm,
let’s go through a detailed example demonstrating how the algorithm
ensures proposer optimality and receiver pessimality.

#### Example Setup

Consider a set of students $`\{S_1, S_2, S_3\}`$ and a set of schools
$`\{C_1, C_2, C_3\}`$ with the following preference lists:

- **Students’ Preferences:**

  - $`S_1`$: $`C_1 > C_2 > C_3`$

  - $`S_2`$: $`C_2 > C_1 > C_3`$

  - $`S_3`$: $`C_1 > C_3 > C_2`$

- **Schools’ Preferences:**

  - $`C_1`$: $`S_2 > S_1 > S_3`$

  - $`C_2`$: $`S_1 > S_3 > S_2`$

  - $`C_3`$: $`S_3 > S_2 > S_1`$

#### Step-by-Step Execution

We will execute the Gale-Shapley algorithm step-by-step to demonstrate
how it produces an optimal matching.

<div class="algorithm">

<div class="algorithmic">

Initialize all students and schools as free $`S_i`$ proposes to the next
school $`C_j`$ on his preference list $`(S_i, C_j)`$ become engaged
$`C_j`$ breaks the engagement with $`S_k`$ $`(S_i, C_j)`$ become engaged
$`S_k`$ becomes free Output the set of engaged pairs $`(S_i, C_j)`$

</div>

</div>

**Execution Steps:**

- **Step 1:** $`S_1`$ proposes to $`C_1`$, and they become engaged.

- **Step 2:** $`S_2`$ proposes to $`C_2`$, and they become engaged.

- **Step 3:** $`S_3`$ proposes to $`C_1`$, but $`C_1`$ prefers $`S_1`$
  over $`S_3`$. So, $`S_3`$ proposes to $`C_3`$, and they become
  engaged.

#### Analysis of the Result

The resulting matching is:

- $`(S_1, C_1)`$

- $`(S_2, C_2)`$

- $`(S_3, C_3)`$

**Proposer Optimality:** Each student ends up with the best possible
school according to their preferences:

- $`S_1`$ gets $`C_1`$, which is his top choice.

- $`S_2`$ gets $`C_2`$, which is his top choice.

- $`S_3`$ gets $`C_3`$, which is his top choice.

**Receiver Pessimality:** Each school gets the least preferred student
among all stable matchings:

- $`C_1`$ gets $`S_1`$, which is better than $`S_3`$ but not as
  preferred as $`S_2`$.

- $`C_2`$ gets $`S_2`$, which is better than $`S_3`$ but not as
  preferred as $`S_1`$.

- $`C_3`$ gets $`S_3`$, which is the least preferred student.

This example demonstrates how the Gale-Shapley algorithm ensures that
the resulting matching is optimal for proposers while still maintaining
stability for receivers.

## Applications and Extensions

Stable matching algorithms have numerous applications and extensions
beyond the basic framework. This section explores various practical
scenarios where stable matching problems are adapted and extended to
suit specific needs. We will discuss some common variants of the stable
matching problem, algorithm modifications, and practical considerations.

### Variants of the Stable Matching Problem

The basic stable matching problem can be extended to address more
complex real-world scenarios. Two well-known variants include the
Hospitals/Residents problem and the College Admissions problem.

#### Hospitals/Residents Problem

The Hospitals/Residents problem is a generalization of the stable
marriage problem. Here, instead of one-to-one matching, we have
one-to-many matching where each hospital can accept multiple residents.

**Problem Definition:**

- **Participants:** A set of hospitals $`H = \{H_1, H_2, \ldots, H_m\}`$
  and a set of residents $`R = \{R_1, R_2, \ldots, R_n\}`$.

- **Preferences:** Each hospital has a preference list over the
  residents and a quota indicating the maximum number of residents it
  can accept. Each resident has a preference list over the hospitals.

- **Objective:** Find a stable matching where no hospital-resident pair
  would prefer each other over their current matches.

**Example:** Consider three hospitals $`H_1, H_2, H_3`$ with quotas 2,
1, and 1, respectively, and four residents $`R_1, R_2, R_3, R_4`$ with
the following preferences:

- **Residents’ Preferences:**

  - $`R_1`$: $`H_1 > H_2 > H_3`$

  - $`R_2`$: $`H_2 > H_3 > H_1`$

  - $`R_3`$: $`H_3 > H_1 > H_2`$

  - $`R_4`$: $`H_1 > H_3 > H_2`$

- **Hospitals’ Preferences:**

  - $`H_1`$: $`R_1 > R_4 > R_3 > R_2`$

  - $`H_2`$: $`R_2 > R_1 > R_3 > R_4`$

  - $`H_3`$: $`R_3 > R_1 > R_2 > R_4`$

Using the Gale-Shapley algorithm, we can determine a stable matching
where hospitals and residents are paired optimally according to their
preferences and quotas.

#### College Admissions Problem

The College Admissions problem is another variant where each college can
accept multiple students, and each student can apply to multiple
colleges.

**Problem Definition:**

- **Participants:** A set of colleges $`C = \{C_1, C_2, \ldots, C_k\}`$
  and a set of students $`S = \{S_1, S_2, \ldots, S_m\}`$.

- **Preferences:** Each college has a preference list over the students
  and a quota. Each student has a preference list over the colleges.

- **Objective:** Find a stable matching where no college-student pair
  would prefer each other over their current matches.

**Example:** Consider three colleges $`C_1, C_2, C_3`$ with quotas 2, 1,
and 1, respectively, and four students $`S_1, S_2, S_3, S_4`$ with the
following preferences:

- **Students’ Preferences:**

  - $`S_1`$: $`C_1 > C_2 > C_3`$

  - $`S_2`$: $`C_2 > C_1 > C_3`$

  - $`S_3`$: $`C_1 > C_3 > C_2`$

  - $`S_4`$: $`C_3 > C_1 > C_2`$

- **Colleges’ Preferences:**

  - $`C_1`$: $`S_1 > S_2 > S_3 > S_4`$

  - $`C_2`$: $`S_2 > S_1 > S_3 > S_4`$

  - $`C_3`$: $`S_3 > S_4 > S_1 > S_2`$

Using a modified version of the Gale-Shapley algorithm, we can determine
a stable matching where colleges and students are paired optimally
according to their preferences and quotas.

### Algorithm Modifications

In practice, stable matching problems often require modifications to the
basic algorithm to account for additional constraints or objectives.
Some common modifications include:

- **Handling Ties:** When participants have indifference among some
  options, special rules are needed to break ties and ensure stability.

- **Weighted Preferences:** Modifying the algorithm to account for
  weighted preferences, where some matches are more desirable than
  others based on additional criteria.

- **Multiple Rounds:** Implementing multiple rounds of matching to allow
  for adjustments and improvements to the initial matches.

### Practical Considerations

Implementing stable matching algorithms in real-world scenarios involves
several practical considerations:

- **Scalability:** Ensuring the algorithm can handle large numbers of
  participants efficiently.

- **Data Privacy:** Protecting the preferences and personal information
  of participants.

- **Legal and Ethical Constraints:** Adhering to legal and ethical
  guidelines, especially in sensitive areas like school admissions and
  job placements.

Understanding these practical considerations is crucial for the
successful application of stable matching algorithms in real-world
scenarios.

## Conclusion

In this section, we conclude our discussion on stable matching
algorithms by summarizing the key concepts and providing recommendations
for further reading and resources.

### Summary of Key Concepts

Stable matching algorithms play a crucial role in various real-world
applications where the goal is to find a matching that is stable and
optimal according to the participants’ preferences. The key concepts
covered in this discussion include:

- **Stable Matching:** The concept where no pair of elements would
  rather be matched with each other than their current partners,
  ensuring stability in the matching process.

- **Gale-Shapley Algorithm:** A fundamental algorithm used to find
  stable matchings in a variety of settings, including the basic stable
  marriage problem and its many variants such as the Hospitals/Residents
  problem and the College Admissions problem.

- **Optimality:** The property of the Gale-Shapley algorithm that
  guarantees an optimal matching for one of the groups, either the
  proposers or the receivers, depending on the implementation.

- **Variants and Extensions:** The adaptation of the stable matching
  framework to address more complex scenarios, such as one-to-many
  matchings in the Hospitals/Residents problem and the College
  Admissions problem.

- **Algorithm Modifications and Practical Considerations:** The
  necessary modifications to the basic algorithm to handle ties,
  weighted preferences, and other real-world constraints, as well as the
  importance of scalability, data privacy, and legal and ethical
  considerations in practical implementations.

### Further Reading and Resources

To deepen your understanding of stable matching algorithms and explore
more advanced topics, we recommend the following resources:

- **Books:**

  - "Algorithm Design" by Jon Kleinberg and Éva Tardos

  - "Introduction to Algorithms" by Thomas H. Cormen, Charles E.
    Leiserson, Ronald L. Rivest, and Clifford Stein

  - "Two-Sided Matching: A Study in Game-Theoretic Modeling and
    Analysis" by Alvin E. Roth and Marilda A. Oliveira Sotomayor

- **Papers:**

  - "College Admissions and the Stability of Marriage" by D. Gale and
    L.S. Shapley

  - "The Theory of Stable Allocations and the Practice of Market Design"
    by Alvin E. Roth

- **Online Tutorials and Courses:**

  - Coursera course on "Algorithms" by Princeton University

  - MIT OpenCourseWare on "Introduction to Algorithms" (6.006)

- **Open-Source Libraries and Algorithm Implementations:**

  - **Python:** The `networkx` library includes implementations of
    various matching algorithms.

  - **Java:** The JGraphT library provides graph and matching algorithm
    implementations.

  - **C++:** The Boost Graph Library offers robust graph data structures
    and algorithms.

These resources will provide a comprehensive understanding of stable
matching algorithms and their applications, offering both theoretical
insights and practical implementations.

## Exercises and Problems

### Conceptual Questions to Test Understanding of Chapter

To ensure a deep understanding of the concepts discussed in the Stable
Matching chapter, answer the following questions:

1.  **Define a stable matching.** What conditions must be satisfied for
    a matching to be considered stable?

2.  **Explain the Gale-Shapley Algorithm.** How does it guarantee a
    stable matching?

3.  **Can there be multiple stable matchings for a given set of
    preferences?** If so, provide an example scenario.

4.  **What are the potential real-world applications of the stable
    matching problem?** Provide at least two examples.

### Practical Exercises

These exercises will help apply the theoretical knowledge of stable
matching to practical problems:

1.  **Implement the Gale-Shapley Algorithm:**

    - Write a program in your preferred programming language to
      implement the Gale-Shapley Algorithm.

    - Test your implementation with the following preference lists:

      - Men’s preferences: $`M_1: [W_1, W_2, W_3]`$,
        $`M_2: [W_2, W_3, W_1]`$, $`M_3: [W_3, W_1, W_2]`$

      - Women’s preferences: $`W_1: [M_3, M_1, M_2]`$,
        $`W_2: [M_1, M_2, M_3]`$, $`W_3: [M_2, M_3, M_1]`$

    - Output the stable matching produced by your implementation.

2.  **Analyze a Matching Scenario:**

    - Given the following preferences, determine if the matching
      $`(M_1, W_2), (M_2, W_3), (M_3, W_1)`$ is stable.

      - Men’s preferences: $`M_1: [W_2, W_3, W_1]`$,
        $`M_2: [W_3, W_1, W_2]`$, $`M_3: [W_1, W_2, W_3]`$

      - Women’s preferences: $`W_1: [M_1, M_2, M_3]`$,
        $`W_2: [M_2, M_3, M_1]`$, $`W_3: [M_3, M_1, M_2]`$

    - Justify your answer by checking for any blocking pairs.

3.  **Modify Preferences:**

    - Modify the preference lists in the previous exercise to create a
      scenario where the given matching is not stable.

    - Describe the changes made and explain why the new matching is not
      stable.

4.  **Explore Variations:**

    - Explore how the Gale-Shapley Algorithm behaves when preferences
      are not strictly ordered, i.e., some individuals have ties in
      their preferences.

    - Implement and test your solution with the following preferences:

      - Men’s preferences: $`M_1: [W_1, W_2/W_3]`$,
        $`M_2: [W_2, W_3, W_1]`$, $`M_3: [W_3, W_1, W_2]`$

      - Women’s preferences: $`W_1: [M_1, M_2/M_3]`$,
        $`W_2: [M_3, M_1, M_2]`$, $`W_3: [M_2, M_3, M_1]`$

    - Discuss the outcome and any challenges faced in implementing this
      variation.
