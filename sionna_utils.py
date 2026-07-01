from pathlib import Path
import numpy as np

def trimesh_to_sionna_scene(
    mesh,
    out_dir,
    *,
    scene_name="sionna_scene",
    object_name="terrain",
    material_name="lunar_soil",
    relative_permittivity=2.87,
    conductivity=0.00133,
    thickness=0.10,
    scattering_coefficient=0.0,
    xpd_coefficient=0.0,
    frequency=2.4e9,
    file_format="ply",
    scale=1.0,
    invert_normals=False,
    cleanup=True,
    load=True,
    merge_shapes=False,
    verbose=True,
):
    """
    Export a trimesh.Trimesh object to a Sionna RT-loadable XML scene.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Your existing mesh object.

    out_dir : str or Path
        Folder where the mesh file and XML scene will be written.

    scene_name : str
        XML filename without extension.

    object_name : str
        Name/id of the mesh object inside Sionna.

    material_name : str
        Name/id of the radio material.

    relative_permittivity : float
        Real relative permittivity epsilon_r.

    conductivity : float
        Conductivity [S/m] used by Sionna.

    thickness : float
        Material thickness [m].

    scattering_coefficient : float
        0.0 means no diffuse scattering.

    xpd_coefficient : float
        Cross-polarization discrimination coefficient.

    frequency : float
        Scene frequency [Hz]. Only applied if load=True.

    file_format : {"ply", "obj"}
        Mesh file format.

    scale : float
        Geometric scale applied before export. Use 1000.0 if your mesh is in km.

    invert_normals : bool
        If True, flips face winding/normals before export.

    cleanup : bool
        If True, removes obvious degenerate/unreferenced mesh elements.

    load : bool
        If True, calls sionna.rt.load_scene and returns the loaded scene.

    merge_shapes : bool
        Passed to sionna.rt.load_scene. Keep False for debugging object names.

    verbose : bool
        Print export/load diagnostics.

    Returns
    -------
    result : dict
        {
            "scene": loaded Sionna scene or None,
            "xml_path": Path,
            "mesh_path": Path,
            "mesh": exported trimesh copy
        }
    """

    if file_format.lower() not in ("ply", "obj"):
        raise ValueError("file_format must be 'ply' or 'obj'.")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_format = file_format.lower()
    mesh_path = out_dir / f"{object_name}.{file_format}"
    xml_path = out_dir / f"{scene_name}.xml"

    # Copy so the original mesh is untouched
    m = mesh.copy()

    if scale != 1.0:
        m.apply_scale(float(scale))

    if invert_normals:
        m.invert()

    if cleanup:
        # Different trimesh versions expose these slightly differently;
        # call only what exists.
        if hasattr(m, "remove_unreferenced_vertices"):
            m.remove_unreferenced_vertices()
        if hasattr(m, "remove_degenerate_faces"):
            m.remove_degenerate_faces()
        if hasattr(m, "remove_duplicate_faces"):
            m.remove_duplicate_faces()
        if hasattr(m, "merge_vertices"):
            m.merge_vertices()

    # Export mesh
    m.export(mesh_path)

    # Use relative filename inside XML, so moving the folder together works.
    mesh_filename = mesh_path.name

    xml = f"""<scene version="3.0.0">

    <bsdf type="radio-material" id="{material_name}">
        <float name="relative_permittivity" value="{relative_permittivity}"/>
        <float name="conductivity" value="{conductivity}"/>
        <float name="thickness" value="{thickness}"/>
        <float name="scattering_coefficient" value="{scattering_coefficient}"/>
        <float name="xpd_coefficient" value="{xpd_coefficient}"/>
    </bsdf>

    <shape type="{file_format}" id="{object_name}">
        <string name="filename" value="{mesh_filename}"/>
        <ref id="{material_name}"/>
    </shape>

</scene>
"""

    xml_path.write_text(xml, encoding="utf-8")

    scene = None
    if load:
        from sionna.rt import load_scene

        scene = load_scene(str(xml_path), merge_shapes=merge_shapes)
        scene.frequency = float(frequency)

    if verbose:
        print("=" * 80)
        print("Sionna scene export")
        print("=" * 80)
        print(f"Mesh file:        {mesh_path}")
        print(f"XML file:         {xml_path}")
        print(f"Object name:      {object_name}")
        print(f"Material name:    {material_name}")
        print(f"eps_r:            {relative_permittivity}")
        print(f"sigma [S/m]:      {conductivity}")
        print(f"thickness [m]:    {thickness}")
        print(f"frequency [Hz]:   {frequency}")
        print(f"num vertices:     {len(m.vertices)}")
        print(f"num faces:        {len(m.faces)}")
        print(f"bounds min:       {np.round(m.bounds[0], 6)}")
        print(f"bounds max:       {np.round(m.bounds[1], 6)}")

        if scene is not None:
            print()
            print("Loaded Sionna scene objects:")
            for name, obj in scene.objects.items():
                mat_name = obj.radio_material.name if obj.radio_material is not None else None
                print(f"  {name} -> material: {mat_name}")

            print()
            print("Loaded Sionna radio materials:")
            for name, mat in scene.radio_materials.items():
                print(f"  {name}")

        print("=" * 80)

    return {
        "scene": scene,
        "xml_path": xml_path,
        "mesh_path": mesh_path,
        "mesh": m,
    }


