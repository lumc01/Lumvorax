# LUM/VORAX - Système de Calcul Basé sur la Présence

## Description

Le projet LUM/VORAX implémente un nouveau paradigme de calcul basé sur des unités de "présence" (LUM) plutôt que sur des bits traditionnels. Ce système utilise des transformations spatiales et des opérations naturelles pour manipuler l'information.

## Concepts Clés

### LUM (Unité de Présence)
- **Présence** : 0 ou 1 (état fondamental)
- **Position spatiale** : coordonnées X, Y
- **Type de structure** : linéaire, circulaire, groupe, nœud
- **Traçabilité** : ID unique et timestamp

### Opérations VORAX
- **⧉ Fusion** : Combiner deux groupes
- **⇅ Split** : Répartir équitablement  
- **⟲ Cycle** : Transformation modulaire
- **→ Flux** : Déplacement entre zones
- **Store/Retrieve** : Gestion mémoire
- **Compress/Expand** : Compression Ω

## Structure du Projet

```
src/
├── lum/                 # Core LUM structures
├── vorax/              # VORAX operations  
├── parser/             # Language parser
├── binary/             # Binary conversion
├── logger/             # Logging system
└── main.c              # Demo principal

examples/               # Exemples VORAX
tests/                  # Tests unitaires
```

## Compilation et Exécution

### Prérequis
- Clang ou GCC
- Make

### Build
```bash
make all        # Compilation complète
make run        # Build et exécution de la démo
make test       # Tests
make clean      # Nettoyage
```

### Utilisation
```bash
./bin/lum_vorax
```

## Exemples

### Code VORAX basique
```vorax
zone A, B, C;
mem buffer;

emit A += 3;           # Émet 3 LUMs dans A
split A -> [B, C];     # Répartit A vers B et C  
move B -> C, 1;        # Déplace 1 LUM de B vers C
store buffer <- C, 1;  # Stocke 1 LUM en mémoire
retrieve buffer -> A;  # Récupère vers A
cycle A % 2;           # Cycle modulo 2
```

### Conversion Binaire ↔ LUM
```c
// Entier vers LUMs
binary_lum_result_t* result = convert_int32_to_lum(42);

// Chaîne binaire vers LUMs  
binary_lum_result_t* result = convert_bits_to_lum("11010110");

// LUMs vers entier
int32_t value = convert_lum_to_int32(lum_group);
```

## Architecture

### Types de Base
- `lum_t` : Unité LUM individuelle
- `lum_group_t` : Collection de LUMs
- `lum_zone_t` : Conteneur spatial
- `lum_memory_t` : Stockage mémoire

### Opérations
- Conservation automatique des LUMs
- Traçabilité complète des transformations
- Vérification d'intégrité
- Log détaillé des opérations

## Philosophie

Le système LUM/VORAX représente un paradigme **post-numérique** où :
- L'information est perçue comme **présence** plutôt que nombre
- Les transformations sont **spatiales** et **naturelles**  
- Le calcul devient **intuitif** et **visuel**
- La conservation des ressources est **garantie**

## Avantages

1. **Sécurité** : Types linéaires empêchent la duplication
2. **Déterminisme** : Même entrée → même sortie  
3. **Parallélisme** : Zones indépendantes sans conflicts
4. **Traçabilité** : Log complet de toutes les opérations
5. **Intuitivité** : Paradigme naturel et visuel

## Status

✅ **Implémentation complète** du système de base
✅ **Parser VORAX** pour le langage
✅ **Conversions binaires** bidirectionnelles  
✅ **Système de log** avancé
✅ **Tests** et exemples fonctionnels

## License

Projet LUM/VORAX - Implémentation C complète