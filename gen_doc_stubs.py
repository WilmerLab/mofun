#!/usr/bin/env python

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

for path in sorted(Path("mofun").glob("*.py")):
    module_path = path.relative_to("").with_suffix("")
    print("XXXXXXX: ", module_path)
    doc_path = path.relative_to("mofun").with_suffix(".md")
    print(doc_path)
    full_doc_path = Path("reference", doc_path)
    print(full_doc_path)

    nav[module_path.parts] = doc_path

    with mkdocs_gen_files.open(full_doc_path, "w") as f:
        ident = ".".join(module_path.parts)
        print("::: " + ident, file=f)

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

nav["mkdocs_autorefs", "references"] = "autorefs/references.md"
nav["mkdocs_autorefs", "plugin"] = "autorefs/plugin.md"

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
