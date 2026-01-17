# Multiplication Matrix-Matrix : OpenMP + MPI Hybride

## Qu'est-ce que ce projet ?

Ce projet implémente la multiplication de deux matrices carrées (2000 x 2000) en utilisant plusieurs approches de parallélisation. L'idée est d'explorer comment utiliser OpenMP pour paralléliser le code sur un seul processeur multi-cœurs et comment utiliser MPI pour distribuer le calcul entre plusieurs processus. Ce projet pédagogique permet de comprendre les performances différentes obtenues selon la stratégie de parallélisation choisie.

## Contenu du Projet

Le projet contient les fichiers suivants :

```
├── 01_sequential.cpp      Version séquentielle (sans parallélisme)
├── 02_openmp.cpp          Version parallélisée avec OpenMP
├── 03_mpi.cpp             Version parallélisée avec MPI
├── 04_hybrid.cpp          Version hybride OpenMP+MPI
├── run_all.sh             Script qui compile et exécute tout
├── plot_results.py        Script qui génère les graphiques
├── RAPPORT.md             Rapport du projet
└── README.md              Ce fichier
```

## Installation sur WSL

Pour installer les dépendances nécessaires sur Windows Subsystem for Linux :

```bash
sudo apt update
sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3-pip
pip3 install pandas matplotlib numpy
```

Vérifier que les outils sont correctement installés :

```bash
g++ --version
mpirun --version
python3 --version
```

## Utilisation

Pour compiler et exécuter le projet complet :

```bash
chmod +x run_all.sh
./run_all.sh
```

Le script va compiler les 4 versions et exécuter les tests. Ensuite, générer les graphiques :

```bash
python3 plot_results.py
```

Cela génère :

- `metrics.csv` : Fichier contenant les temps d'exécution et GFLOPS
- `performance_analysis.png` : Graphiques des résultats
- `performance_report.txt` : Rapport d'analyse automatique

## Spécifications Techniques

Le projet travaille sur des matrices de dimensions :

- Matrice A : 2000 x 2000 éléments flottants
- Matrice B : 2000 x 2000 éléments flottants
- Matrice C : 2000 x 2000 éléments flottants (résultat)
- Taille totale en mémoire : environ 96 MB
- Nombre d'opérations : environ 16 milliards de multiplications

## Approches Implémentées

### 1. Version Séquentielle

La version de base sans parallélisme sert de référence. Elle exécute le calcul sur un seul cœur.

### 2. Version OpenMP

OpenMP permet de paralléliser les boucles sur plusieurs threads d'un même processus. Cette version teste automatiquement entre 1 et 8 threads pour mesurer l'impact du nombre de threads.

Implémentation :

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

La directive `collapse(2)` parallélise les deux boucles imbriquées pour une meilleure distribution du travail. `schedule(dynamic, 32)` utilise un scheduling dynamique avec chunks de 32 itérations.

### 3. Version MPI

MPI distribue le calcul entre plusieurs processus (ranks). Chaque rank traite un sous-ensemble des lignes de la matrice A.

Stratégie :

- Scatter : Chaque rank reçoit les lignes correspondantes de la matrice A
- Broadcast : Tous les ranks reçoivent une copie complète de la matrice B
- Calcul local : Chaque rank calcule ses lignes en parallèle
- Gather : Les résultats sont rassemblés dans la matrice C

### 4. Version Hybride

La version hybride combine OpenMP et MPI. MPI distribue le travail entre les processus, puis OpenMP parallélise le calcul à l'intérieur de chaque processus.

## Configuration de l'Environnement de Test

Les tests ont été réalisés sur un système avec les caractéristiques suivantes :

- Processeur : 8 cores logiques
- Mémoire RAM : 8 GB (limitation qui affecte la performance)
- Système : Windows 11 avec WSL Ubuntu 20.04
- Compilateur : g++ 9.x avec optimisations -O3 -march=native
- OpenMPI : Version 3.1.3

Note importante : La RAM disponible est une limite importante. Avec 8 GB seulement, les matrices de 2000x2000 consomment 96 MB chacune, ce qui laisse peu de place pour le cache et les opérations temporaires. Cela explique pourquoi les performances observées sont plus faibles que sur une machine avec plus de RAM.

## Résultats Observés

Les résultats montrent les performances de toutes les approches de parallélisation :

| Configuration | Temps (ms) | GFLOPS | Speedup |
|---|---|---|---|
| Séquentiel | 53902.2 | 0.297 | 1.00x |
| OpenMP 1T | 74568.6 | 0.215 | 0.72x |
| OpenMP 2T | 41240.6 | 0.388 | 1.31x |
| OpenMP 3T | 29685.9 | 0.539 | 1.82x |
| OpenMP 4T | 26680.6 | 0.600 | 2.02x |
| OpenMP 5T | 25845.7 | 0.619 | 2.09x |
| OpenMP 6T | 24029.4 | 0.666 | 2.24x |
| OpenMP 7T | 25364.7 | 0.631 | 2.13x |
| OpenMP 8T | 27079.5 | 0.591 | 1.99x |
| MPI 2 Ranks | 30322.8 | 0.528 | 1.78x |
| MPI 4 Ranks | 20352.5 | 0.786 | 2.65x |
| Hybride 2R 1T | 30798.9 | 0.519 | 1.75x |
| Hybride 2R 2T | 19226.6 | 0.832 | 2.80x |

Le speedup maximal global de 2.80x est atteint avec la version hybride (2 ranks avec 2 threads par rank). OpenMP seul atteint 2.24x avec 6 threads. MPI avec 4 ranks atteint 2.65x. La version hybride démontre comment combiner les deux approches pour obtenir les meilleures performances.

## Graphiques de Performance

Le script `plot_results.py` génère quatre graphiques :

1. **Temps d'exécution** : Montre le temps nécessaire pour chaque configuration
2. **GFLOPS** : Affiche la performance en milliards d'opérations flottantes par seconde
3. **Speedup** : Montre le gain relatif par rapport à la version séquentielle
4. **Scaling OpenMP** : Graphique double montrant comment le temps et les GFLOPS varient avec le nombre de threads

## Analyse des Résultats

### Speedup et Scalabilité

Le speedup obtenu (2.24x avec 6 threads) montre que OpenMP parvient à paralléliser efficacement le calcul, mais avec des rendements décroissants passé 6 threads. Cela est normal et dû à plusieurs facteurs :

1. Overhead de threading : Créer et synchroniser des threads a un coût
2. Contention sur le cache L3 : Avec 8 cores partageant le même L3, l'accès mémoire devient un goulot
3. Limite WSL : Exécuter sur WSL plutôt que Linux natif ajoute du surcoût
4. Limite RAM : 8 GB de RAM limite la performance globale

### Performance Absolue

Les performances en GFLOPS sont modestes (0.666 GFLOPS au maximum). Cela est attendu pour plusieurs raisons :

1. WSL a un surcoût par rapport à Linux natif
2. Les matrices 2000x2000 sont trop grandes pour profiter pleinement du cache
3. Pas d'optimisation par tiling/blocking dans le code
4. La limitation RAM force de nombreux accès mémoire lents

## Intérêt Pédagogique du Projet

Ce projet permet de comprendre :

1. Comment utiliser les directives OpenMP pour paralléliser des boucles
2. Comment MPI distribue le travail entre plusieurs processus
3. Comment mesurer et analyser les performances d'un code parallèle
4. Les limitations pratiques de la parallélisation (overhead, cache, memory)
5. Comment les choix d'implémentation affectent les performances

Les résultats montrent que la parallélisation n'est pas magique : elle apporte des gains, mais avec des rendements décroissants et des coûts qui doivent être pris en compte.

## Mise en Œuvre et Points Clés

### Directives OpenMP Utilisées

- `#pragma omp parallel for` : Parallélise une boucle
- `collapse(2)` : Fusionne deux boucles imbriquées pour une meilleure parallélisation
- `schedule(dynamic, 32)` : Utilise un scheduling dynamique avec chunks de 32 itérations, ce qui améliore l'équilibrage du travail

### Appels MPI Utilisés

- `MPI_Init_thread` : Initialise MPI avec support des threads
- `MPI_Scatter` : Distribue les données du rank 0 aux autres ranks
- `MPI_Bcast` : Envoie les données depuis le rank 0 à tous les autres
- `MPI_Gather` : Rassemble les résultats de tous les ranks au rank 0

### Synchronisation

Pour éviter les race conditions en OpenMP, seules les opérations indépendantes sont parallélisées. Chaque thread calcule un sous-ensemble des éléments de la matrice C sans partager de données modifiées.

Pour MPI, les appels de communication (Scatter, Gather, Bcast) incluent une synchronisation implicite qui assure que tous les ranks sont au même point avant de continuer.

## Limitations et Améliorations Possibles

Plusieurs optimisations pourraient améliorer les performances :

1. **Tiling/Blocking** : Diviser les matrices en blocs plus petits pour meilleure localité cache
2. **Affinity** : Lier les threads à des cores spécifiques avec `OMP_PROC_BIND`
3. **Compilation native** : Tester sur Linux natif au lieu de WSL
4. **Matrices plus petites** : Utiliser 1000x1000 au lieu de 2000x2000 pour mieux profiter du cache
5. **Algorithmes plus avancés** : Utiliser Strassen ou autres pour matrices plus grandes

Cependant, pour une démonstration pédagogique, le code actuel est suffisant et fonctionne correctement.

## Fichiers Générés

Après l'exécution, les fichiers suivants sont créés :

- `metrics.csv` : Tableau des résultats (temps, GFLOPS, throughput)
- `performance_analysis.png` : Quatre graphiques de performance
- `performance_report.txt` : Rapport détaillé généré automatiquement

## Conclusion

Ce projet démontre de manière pratique comment utiliser OpenMP, MPI et une approche hybride pour paralléliser un calcul intensif. Les résultats obtenus montrent que :

1. **OpenMP seul** : Speedup de 2.24x avec 6 threads
2. **MPI seul** : Speedup de 2.65x avec 4 processus
3. **Hybride** : Speedup maximal de 2.80x avec 2 ranks et 2 threads par rank

La version hybride offre les meilleures performances en combinant la distribution entre processus (MPI) et la parallélisation locale avec threads (OpenMP). Cela démontre l'intérêt d'une approche hybride pour exploiter tous les niveaux de parallélisme disponibles sur un système moderne.

Les limitations dues à WSL et à la RAM disponible (8 GB) expliquent pourquoi les performances absolues (0.832 GFLOPS au maximum) sont modestes. Sur une machine native Linux avec plus de RAM, les résultats seraient meilleurs. Cependant, cette implémentation démontre correctement les concepts fondamentaux de la parallélisation et comment mesurer et comparer les performances de différentes approches.