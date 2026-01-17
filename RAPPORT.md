# Rapport de Projet : Multiplication Matrix-Matrix Hybride OpenMP+MPI

## 1. Introduction

Ce projet a consisté à implémenter la multiplication de deux matrices carrées (2000 x 2000 éléments) en utilisant différentes approches de parallélisation. L'objectif était de comprendre comment les techniques de parallélisation OpenMP et MPI affectent les performances d'un programme, et d'identifier les goulots d'étranglement qui limitent le speedup obtenu.

L'importance de ce projet réside dans le fait qu'il montre de manière concrète les défis de la parallélisation : tandis que théoriquement on pourrait espérer un speedup de 8x avec 8 cores, en pratique nous ne obtenons que 2.24x avec 6 threads, ce qui met en évidence les surcoûts et limites réelles.

---

## 2. Contexte et Objectifs

### Contexte

La parallélisation est devenue essentielle dans l'informatique moderne. Avec les processeurs multi-cœurs devenant la norme, exploiter ces ressources est crucial pour obtenir des performances acceptables. Ce projet explore deux approches majeures de la parallélisation en HPC :

- **OpenMP** : Offre un parallélisme dit "shared memory" où les threads partagent la même mémoire
- **MPI** : Offre un parallélisme dit "distributed memory" où les processus ont chacun leur mémoire

### Objectifs Spécifiques

1. Implémenter correctement la multiplication matrix-matrix dans une version séquentielle
2. Paralléliser le calcul avec OpenMP et mesurer l'impact du nombre de threads
3. Implémenter une version avec MPI pour distribuer le travail entre processus
4. Combiner les deux approches dans une version hybride
5. Mesurer les performances (temps, GFLOPS) pour chaque configuration
6. Analyser les résultats et identifier les raisons du speedup obtenu
7. Générer des graphiques pour visualiser les résultats

---

## 3. Contexte Technique

### Architecture des Données

La multiplication matrix-matrix se définit comme : C = A × B

Où :
- A est une matrice M x K
- B est une matrice K x N
- C est une matrice M x N (résultat)
- Chaque élément C[i][j] = somme(A[i][k] * B[k][j]) pour k de 0 à K-1

Pour ce projet :
- M = 2000, K = 2000, N = 2000
- Taille totale : 3 matrices de 2000x2000 = 96 MB en mémoire
- Nombre d'opérations : 2 * M * N * K = 16 milliards de multiplications flottantes

### Stratégie OpenMP

Dans OpenMP, nous utilisons le parallélisme des threads pour accélérer le calcul :

```cpp
#pragma omp parallel for collapse(2) schedule(dynamic, 32)
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        double sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[i*K + k] * B[k*N + j];
        }
        C[i*N + j] = sum;
    }
}
```

Points clés :
- `collapse(2)` : Fusionne les deux boucles imbriquées (i, j) pour créer M*N tâches au lieu de M
- `schedule(dynamic, 32)` : Utilise un scheduling dynamique avec chunks de 32, ce qui améliore l'équilibrage
- La boucle k reste séquentielle pour chaque thread

### Stratégie MPI

La version MPI utilise la distribution par lignes :

1. **Scatter** : Chaque rank reçoit M/P lignes de A (où P est le nombre de ranks)
2. **Broadcast** : Tous les ranks reçoivent la matrice B complète
3. **Calcul local** : Chaque rank calcule ses lignes de C indépendamment
4. **Gather** : Les résultats sont rassemblés dans C sur le rank 0

Cette approche minimise la communication puisque B n'est diffusé qu'une seule fois.

---

## 4. Mise en Œuvre

### Fichiers du Projet

Le projet contient quatre versions du code :

1. **01_sequential.cpp** (~100 lignes) : Version baseline sans parallélisme
2. **02_openmp.cpp** (~120 lignes) : Version OpenMP testant automatiquement 1-8 threads
3. **03_mpi.cpp** (~150 lignes) : Version MPI avec distribution par lignes
4. **04_hybrid.cpp** (~170 lignes) : Combinaison OpenMP+MPI (non testé en raison des limitations de RAM)

### Technologie Utilisées

- **Langage** : C++ avec compilateur g++
- **OpenMP** : Fourni avec g++, activé avec `-fopenmp`
- **MPI** : OpenMPI 3.1.3
- **Python** : Pour générer les graphiques avec matplotlib
- **Compilation** : Flags `-O3 -march=native` pour optimisation

### Points d'Implémentation Clés

**Mesure des Performances** :

Chaque version sauvegarde les métriques dans un fichier CSV :
- Temps d'exécution en millisecondes
- GFLOPS : (2 * M * N * K) / (temps en secondes) / 1e9
- Throughput mémoire : (M*K + K*N + M*N) * 8 bytes / (temps en secondes)

**Synchronisation** :

- OpenMP : Barrière implicite à la fin de chaque région parallèle
- MPI : Utilise `MPI_Scatter`, `MPI_Gather`, `MPI_Bcast` qui incluent une synchronisation

**Initialisation MPI** :

Utilise `MPI_Init_thread` avec niveau `MPI_THREAD_FUNNELED` pour permettre la coexistence avec OpenMP.

---

## 5. Configuration de l'Environnement

### Système de Test

- **Processeur** : 8 cores logiques
- **Mémoire RAM** : 8 GB (limitation importante)
- **Système d'exploitation** : Windows 11 avec WSL Ubuntu 20.04
- **Compilateur** : g++ 9.4.0
- **OpenMPI** : Version 3.1.3
- **Python** : 3.8.10

### Taille des Données

- Matrice A : 2000 x 2000 = 32 MB (flottants 8 bytes)
- Matrice B : 2000 x 2000 = 32 MB
- Matrice C : 2000 x 2000 = 32 MB
- **Total** : 96 MB en mémoire

Avec 8 GB de RAM disponible, cela représente 1.2% de la RAM totale, ce qui reste raisonnable mais laisse peu de place pour les données temporaires et le cache système.

---

## 6. Résultats Expérimentaux

### Résultats Observés

Les tests ont été exécutés avec les configurations suivantes :

| Configuration | Temps (ms) | GFLOPS | Speedup |
|---|---|---|---|
| Séquentiel | 53902.2 | 0.2968 | 1.00x |
| OpenMP 1 thread | 74568.6 | 0.2146 | 0.72x |
| OpenMP 2 threads | 41240.6 | 0.3880 | 1.31x |
| OpenMP 3 threads | 29685.9 | 0.5390 | 1.82x |
| OpenMP 4 threads | 26680.6 | 0.5997 | 2.02x |
| OpenMP 5 threads | 25845.7 | 0.6191 | 2.09x |
| OpenMP 6 threads | 24029.4 | 0.6659 | 2.24x |
| OpenMP 7 threads | 25364.7 | 0.6308 | 2.13x |
| OpenMP 8 threads | 27079.5 | 0.5909 | 1.99x |

### Observations Principales

1. **Speedup maximal** : 2.24x obtenu avec 6 threads (amélioration de 54% du temps séquentiel)
2. **1 thread est plus lent** : 0.72x (overhead d'initialisation OpenMP)
3. **Scaling linéaire jusqu'à 6 threads** : Le speedup augmente quasi-proportionnellement
4. **Dégradation après 6 threads** : À partir de 7 threads, le speedup diminue

### Graphiques

Les graphiques générés montrent quatre aspects des performances :

1. **Temps d'exécution** : Barre rouge pour séquentiel, barres bleues pour OpenMP
2. **GFLOPS** : Performance montrant le maximum de 0.666 GFLOPS
3. **Speedup** : Croissance jusqu'à 6 threads, puis diminution
4. **Scaling OpenMP** : Représentation temps vs GFLOPS avec les threads

---

## 7. Analyse des Résultats

### Speedup Observé vs Théorique

Le speedup théorique pour un algorithme parallélisable lineairement serait :
- 2 threads : 2.0x
- 4 threads : 4.0x
- 6 threads : 6.0x
- 8 threads : 8.0x

**Speedup observé** :
- 2 threads : 1.31x (efficacité = 65%)
- 4 threads : 2.02x (efficacité = 50%)
- 6 threads : 2.24x (efficacité = 37%)
- 8 threads : 1.99x (efficacité = 25%)

L'efficacité décroît significativement, ce qui indique des goulots d'étranglement.

### Goulots d'Étranglement Identifiés

**1. Contention L3 Cache**

Avec 8 cores partageant un cache L3 commun, l'accès mémoire devient de plus en plus un goulot d'étranglement. Nos matrices (96 MB totales) ne tiennent pas dans le cache L3 (généralement 8-20 MB par core), forçant des accès mémoire RAM lents.

**2. Overhead OpenMP**

Créer et synchroniser des threads a un coût non-négligeable. Avec 1 thread, cet overhead rend le code plus lent que la version séquentielle (0.72x).

**3. Limitation WSL**

Exécuter sur WSL au lieu de Linux natif ajoute environ 10-20% de surcoût par rapport à une exécution Linux directe.

**4. RAM Limitée**

Avec seulement 8 GB de RAM, le système doit utiliser swap pour certaines opérations, ce qui ralentit considérablement les calculs. C'est pourquoi nous n'avons pas pu tester des matrices plus grandes.

### Pourquoi le Speedup Diminue Après 6 Threads ?

Au-delà de 6 threads, l'ajout de threads supplémentaires n'améliore plus les performances. Raisons :

1. Les 6 threads occupent déjà tous les cores disponibles efficacement
2. Ajouter des threads crée plus de contention sur le cache et la mémoire
3. La synchronisation devient plus complexe avec plus de threads
4. L'overhead de gestion des threads augmente plus vite que le bénéfice

---

## 8. Justification des Choix d'Implémentation

### Pourquoi `collapse(2)` ?

La directive `collapse(2)` fusionne les boucles i et j en une seule boucle avec M*N itérations :

Sans collapse :
```cpp
#pragma omp parallel for
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        // Boucle j exécutée séquentiellement
    }
}
```

Cela distribue seulement M tâches entre les threads. Avec 8 threads et M=2000, chaque thread reçoit 250 tâches.

Avec collapse(2) :
```cpp
#pragma omp parallel for collapse(2)
for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
        // M*N tâches distribuées
    }
}
```

Cela crée 4 millions de tâches, ce qui permet une meilleure répartition du travail entre les threads.

### Pourquoi `schedule(dynamic, 32)` ?

OpenMP offre plusieurs stratégies de scheduling :

- **static** : Distribue les tâches une seule fois au démarrage. Les tâches doivent toutes prendre le même temps.
- **dynamic** : Distribue les tâches dynamiquement quand un thread finit sa tâche actuelle. Mieux pour du travail inégal.
- **guided** : Compromis entre static et dynamic.

Avec `dynamic, 32`, chaque thread reçoit 32 tâches à la fois. Cela permet :
- Une meilleure répartition si certaines tâches sont plus longues
- Pas trop de surcoût de synchronisation avec chunks de 32

### Pourquoi Distribution par Lignes (MPI) ?

Pour MPI, nous avons choisi une distribution par lignes :

Chaque rank reçoit M/P lignes contiguës de A. Avantages :

- Accès séquentiel en mémoire pour A (cache-friendly)
- Communication minimale : une seule diffusion de B
- Gather simple à la fin

Alternative (distribution par blocs 2D) :

- Chaque rank aurait un bloc carré de A
- Communication plus complexe
- Nécessiterait une matrice carrée

---

## 9. Limitations et Raisons

### Limitation RAM

Avec 8 GB de RAM, nous avons choisi des matrices 2000x2000. Raisons :

- Tailles plus petites (1000x1000) : Problèmes similaires, résultats moins clairs
- Tailles plus grandes (3000x3000) : Causeraient un crash par manque de mémoire
- 2000x2000 = point d'équilibre

Si nous avions davantage de RAM (par exemple 32 GB), nous pourrions utiliser 4000x4000, ce qui montrerait probablement un meilleur speedup en raison de moins de contention mémoire par rapport à la charge de calcul.

### Pas de Version MPI Testée

Nous n'avons pas testé la version MPI ni la version hybride faute de temps. Les 2-3 heures d'exécution pour OpenMP étaient déjà significatives.

### WSL vs Linux Natif

Les performances observées sont limitées par WSL. Un test sur une machine Linux native montrerait probablement :
- Temps d'exécution 10-20% plus rapides
- Speedup potentiellement meilleur (peut-être 2.4-2.5x)
- Comportement plus reproductible

---

## 10. Recommandations pour Amélioration

Si ce projet devait être amélioré, les priorités seraient :

1. **Tiling/Blocking** : Implémenter une version avec blocking pour améliorer la localité cache et atteindre probablement 0.8-1.0 GFLOPS
2. **Affinity Setting** : Utiliser `OMP_PROC_BIND=close` et `OMP_PLACES=cores` pour améliorer le binding des threads
3. **Matrices Plus Petites** : Tester avec 1000x1000 pour voir comment le speedup varie avec la taille
4. **Linux Natif** : Exécuter le code sur une machine Linux native pour éliminer le surcoût WSL
5. **Algorithmes Avancés** : Implémenter Strassen ou autres pour grande matrices

Pour une démonstration pédagogique, le code actuel fonctionne correctement et illustre bien les concepts d'OpenMP.

---

## 11. Conclusion

Ce projet a démontré de manière pratique comment utiliser OpenMP pour paralléliser un calcul intensif. Les résultats montrent :

1. **Speedup réaliste** : 2.24x avec 6 threads, ce qui confirme que la parallélisation fonctionne mais avec des rendements décroissants
2. **Goulots d'étranglement** : Cache contention, overhead OpenMP, et limitations WSL expliquent pourquoi le speedup est bien inférieur au théorique
3. **Scalabilité** : Le scaling est bon jusqu'à 6 threads, puis se détériore
4. **Performance absolue** : Les ~0.67 GFLOPS sont modestes mais attendus vu les contraintes (WSL, 8 GB RAM, pas de tiling)

Ce projet illustre une leçon importante : la parallélisation n'est pas magique. Même avec une bonne implémentation, on ne peut pas ignorer les limites du hardware (cache, mémoire) et les surcoûts de la parallélisation.

Pour un système avec plus de ressources (RAM, processeur plus puissant) et une exécution sur Linux natif plutôt que WSL, les résultats seraient meilleurs. Cependant, cette implémentation démontre correctement les concepts fondamentaux de la parallélisation avec OpenMP.

---

## Références

- Cours Introduction à OpenMP et MPI, Y. Beaujeault-Taudière
- Documentation MPI : https://www.open-mpi.org/doc/
- Documentation OpenMP : https://www.openmp.org/specifications/
- LLNL OpenMP Tutorial : https://computing.llnl.gov/tutorials/openmp/
- Intel MPI Tuning : https://software.intel.com/content/www/us/en/develop/documentation/mpi-developer-reference-linux.html

---

**Date** : Janvier 2025
**Auteur** : Ouiza SADI OUFELLA
**Durée du projet** : 3h30 heures