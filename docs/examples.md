# Examples


## Example 1: functionalizing a MOF

For this example, we will take the UiO-66 MOF and functionalize its linker with a hydroxyl group. We have provided all
the files for this example, but if you were do this procedure on your own structure, you would need to take the
following steps:

* Prepare the structure file as a P1 CIF or a P1 lammps data file.
* Prepare the search pattern. In this example, we are searching for the linker of UiO-66, which is a biphenyl linker. We
  used Vesta to pick one linker in the structure, deleted all other atoms, then exported to a file format that Avogadro
  can read. We opened the file in Avogadro and saved as CML.
* Prepare the replacement pattern. The replacement pattern needs to lie in the same coordinate system as the search
  pattern. The easiest way to do this is to start with the search pattern and simply not move any of the atoms unless
  you want to move them with the replacement operation. For this example, we took the search pattern CML, replaced one
  of the hydrogens on the biphenyl group with an oxygen atom, and added the attached hydrogen to make the hydroxyl. We
  used avogadro's "fix atoms" feature to fix all the atoms except for the newly added ones, then ran optimize structure
  to let the OH group find a more appropriate position. (Note that if you do not fix ALL the atoms except for the
  hydroxyls, many of the atoms will move when you optimize and the atoms of your replacement pattern may insert into an
  odd position!)


<figure markdown>
  ![UiO-66 Linker](img/examples/uio66-linker.png)
  <figcaption>Search pattern: UiO-66 linker</figcaption>
</figure>

<figure markdown>
  ![UiO-66 Linker with Hydroxyl](img/examples/uio66-linker-hydroxyl.png)
  <figcaption>Replacement pattern: UiO-66 linker with hydroxyl functional group</figcaption>
</figure>


Once you have the files prepared, the find / replace operation is very simple. In Python:

```python
from mofun import Atoms, replace_pattern_in_structure

structure = Atoms.load("uio66.cif")
uio66_linker = Atoms.load("uio66-linker.cml")
uio66_linker_w_hydroxyl = Atoms.load("uio66-linker-w-hydroxyl.cml")

structure_w_hdyroxyl = replace_pattern_in_structure(structure, uio66_linker, uio66_linker_w_hydroxyl)

structure_w_hdyroxyl.save("uio66-w-hydroxyl.lmpdat")
structure_w_hdyroxyl.to_ase().write("uio66-w-hydroxyl.cif")
```

In your shell:

```shell
mofun uio66.cif uio66-w-hydroxyl.cif -f uio66-linker.cml -r uio66-linker-w-hydroxyl.cml
```

If you look in the output uio66-w-hydroxyl.cif file, you will see the hydroxyls on all the linkers.

## Example 2: introducing defects into a MOF

For this example, we will introduce defects into UiO-66 by randomly removing linkers from the structure. We will first
replicate the structure to a 2x2x2 so it fulfills minimum image conventions. We do this before adding defects, so that
the defects aren't repeated in the structure. We will create defects for 10%, and 90% of all linkers. (Clearly, having 90% of linkers
be defective would create a non-viable structure, but it is easier to visualize).

As in example 1, we will need to prepare the structure file, search pattern and replacement pattern. See above for discussion
on how we typically do that. We can use the structure and search pattern files from example 1, but we will need a replacement pattern
where the biphenyl ring is removed and there are formate caps where the linker would attach to the metal center.

<figure markdown>
  ![UiO-66 defective linker](img/examples/uio66-linker-defective.png)
  <figcaption>Replacement pattern: UiO-66 defective linker</figcaption>
</figure>


In Python:

```python
from mofun import Atoms, replace_pattern_in_structure

structure = Atoms.load("uio66.cif").replicate((2,2,2))
uio66_linker = Atoms.load("uio66-linker.cml")
uio66_linker_defective = Atoms.load("uio66-linker-defective.cml")

defective10 = replace_pattern_in_structure(structure, uio66_linker, uio66_linker_defective, replace_fraction=0.10)
defective10.to_ase().write("uio66-defective-10.cif")

defective90 = replace_pattern_in_structure(structure, uio66_linker, uio66_linker_defective, replace_fraction=0.90)
defective90.to_ase().write("uio66-defective-90.cif")
```

In your shell:

```shell
mofun uio66.cif uio66-defective-10.cif -f uio66-linker.cml -r uio66-linker-defective.cml --replicate 2 2 2 --replace-fraction=0.10
mofun uio66.cif uio66-defective-90.cif -f uio66-linker.cml -r uio66-linker-defective.cml --replicate 2 2 2 --replace-fraction=0.90
```


## Example 3: parameterizing a replicated MOF using only parameterized linker and metal centers

For this example, we start with an unparameterized CIF file, and then we find and replace both the linker and the metal center
with parameterized versions, thus parameterizing the full structure across periodic boundaries.


## Example 4: searching the CoRE database for UIO-66 analogs

TODO


