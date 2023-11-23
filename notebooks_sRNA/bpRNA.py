#####################################################################
# Rewritten in Python by Google Bard + Olivia Gallup
#####################################################################


import networkx as nx
import numpy as np
import re

#####################################################################
# 2018  Padideh Danaee,Michelle Wiley, Mason Rouches, David Hendrix #
# http://hendrixlab.cgrb.oregonstate.edu                            #
# ----------------------------------------------------------------- #              
# MAIN                                                              #
#####################################################################

DEBUG = 0
USAGE = "Usage:\n$0 <bpseq file or dot-bracket file> \n"

ALL_STRUCTURE_TYPES = ["S", "H", "B", "I", "M", "X", "E", "PK", "PKBP", "NCBP", "SEGMENTS"]


###############
# SUBROUTINES #
###############

def print_structure_types(id, seq, dotbracket, s, k, structure_types, page_number, warnings):
    s = "".join(s)
    k = "".join(k)
    st_file = f"{id}.st"

    with open(st_file, "w") as stf:
        stf.write(f"#Name: {id}\n")
        stf.write(f"#Length: {len(seq)}\n")
        stf.write(f"#PageNumber: {page_number}\n")
        stf.write(warnings)  # warning contains label and newline
        stf.write(f"{seq}\n{dotbracket}\n{s}\n{k}\n")

        for item in structure_types["S"]:
            stf.write(f"{item}\n")

        for item in structure_types["H"]:
            stf.write(f"{item}\n")

        for item in structure_types["B"]:
            stf.write(f"{item}\n")

        for i in sorted(structure_types["I"].keys()):
            for item in structure_types["I"][i]:
                stf.write(f"{item}\n")

        for m in sorted(structure_types["M"].keys()):
            for item in structure_types["M"][m]:
                stf.write(f"{item}\n")

        for item in structure_types["X"]:
            stf.write(f"{item}\n")

        for item in structure_types["E"]:
            stf.write(f"{item}\n")

        for item in structure_types["PK"]:
            stf.write(f"{item}\n")

        for item in structure_types["PKBP"]:
            stf.write(f"{item}\n")

        for item in structure_types["NCBP"]:
            stf.write(f"{item}\n")

        for item in structure_types["SEGMENTS"]:
            stf.write(f"{item}\n")

def build_structure_map(segments, knots, bp, seq):
    G, edges = build_segment_graph(seq, bp, segments, knots)
    dotbracket, page_number = compute_dot_bracket(segments, knots, seq)
    s, pk = compute_structure_array(dotbracket, bp, seq)

    edges_by_type = {}
    pk_loops = {}
    regions = {}
    structure_types = {s: [] for s in ALL_STRUCTURE_TYPES}
    structure_types["I"] = {}
    structure_types["M"] = {}

    for edge in edges:
        # u and v are the segment IDs
        u, v, s1_pos, s2_pos, label = edge
        # the start and stop of edge is adjacent to segment:
        l_start = s1_pos + 1
        l_stop = s2_pos - 1
        e_type = "M"
        if u == v: e_type = "H" 

        if e_type not in edges_by_type: edges_by_type[e_type] = []
        edges_by_type[e_type].append((l_start, l_stop))

    # First, handle Multiloops
    if "M" in edges_by_type:
        mG = nx.DiGraph()  # a directed graph
        for m, edge in enumerate(edges_by_type["M"]):
            # 1-based coords
            m_start, m_stop = edge
            m_seq = seq[m_start - 1:m_stop - m_start + 1]

            # this is important for X regions not adjacent to any other X regions.
            mG.add_node(m)

            for n, other_edge in enumerate(edges_by_type["M"]):
                # 1-based coords
                n_start, n_stop = other_edge
                n_seq = seq[n_start - 1:n_stop - n_start + 1]

                if loop_linked(m_start, m_stop, n_start, n_stop, bp):
                    mG.add_edge(m, n)

        mUG = mG.to_undirected()
        mCC = list(nx.connected_components(mUG))

        multi_loops = []
        external_loops = []

        # create a sorted list of multiloops.
        for c in mCC:
            if is_multiloop(c, mG, edges_by_type):
                c = sorted(
                    c, key=lambda i: edges_by_type["M"][i][0]
                )  # sort branches by position.
                multi_loops.append(c)
            else:
                c = sorted(
                    c, key=lambda i: edges_by_type["M"][i][0]
                )  # sort branches by position.
                external_loops.append(c)


        multi_loops = sorted(multi_loops, key=lambda x: edges_by_type["M"][x[0]][0])
        m_count = 0
        for loop in multi_loops:
            m_count += 1
            mp = 0
            for m in loop:
                mp += 1
                m_start, m_stop = edges_by_type["M"][m]
                m_length = m_stop - m_start + 1
                m_seq = seq[m_start - 1:m_start + m_length - 1]

                bp5_pos1 = m_start - 1
                nuc5_1 = seq[bp5_pos1 - 1:bp5_pos1]
                bp5_pos2 = bp[bp5_pos1]
                nuc5_2 = seq[bp5_pos2 - 1:bp5_pos2]

                bp3_pos1 = m_stop + 1
                nuc3_1 = seq[bp3_pos1 - 1:bp3_pos1]
                bp3_pos2 = bp[bp3_pos1]
                nuc3_2 = seq[bp3_pos2 - 1:bp3_pos2]

                mknots = includes_knot(m_start, m_stop, knots)
                pk = mknots if mknots else ""

                structure_types["M"].setdefault(m_count, []).append(
                    f"M{m_count}.{mp} {m_start}..{m_stop} \"{m_seq}\" ({bp5_pos1},{bp5_pos2}) {nuc5_1}:{nuc5_2} ({bp3_pos1},{bp3_pos2}) {nuc3_1}:{nuc3_2} {pk}\n"
                )

                for k in mknots:
                    pk_loops[k].append([f"M{m_count}.{mp}", m_start, m_stop])

                # Update multiloops in regions
                regions["M"].append([m_start, m_stop])

                # Update base pairing status
		        # if this inequality doesn't hold, it's a branch of length 0.
                if m_start <= m_stop:
                    for i in range(m_start - 1, m_stop):
                        s[i] = "M"

        for loop in external_loops:
            # these need to be classified as "X" for unpaired
            x_count = 0
            loop = sorted(loop, key=lambda x: edges_by_type["M"][x][0])
            for x in loop:
                x_start, x_stop = edges_by_type["M"][x]
                x_seq = seq[x_start - 1:x_stop]

                bp5_pos1 = x_start - 1
                nuc5_1 = seq[bp5_pos1 - 1:bp5_pos1]
                bp5_pos2 = bp[bp5_pos1]
                nuc5_2 = seq[bp5_pos2 - 1:bp5_pos2]

                bp3_pos1 = x_stop + 1
                nuc3_1 = seq[bp3_pos1 - 1:bp3_pos1]
                bp3_pos2 = bp[bp3_pos1]
                nuc3_2 = seq[bp3_pos2 - 1:bp3_pos2]

                xknots = includes_knot(x_start, x_stop, knots)
                pk = xknots if xknots else ""

                if x_start <= x_stop:
                    x_count += 1
                    structure_types["X"].append(
                        f"X{x_count} {x_start}..{x_stop} \"{x_seq}\" ({bp5_pos1},{bp5_pos2}) {nuc5_1}:{nuc5_2} ({bp3_pos1},{bp3_pos2}) {nuc3_1}:{nuc3_2} {pk}\n"
                    )

                for k in xknots:
                    pk_loops[k].append([f"X{x_count}", x_start, x_stop])

                # Check if in a multiloop part or something other than U/X
                for i in range(x_start - 1, x_stop):
                    assert s[i] in ["u", "X"]

    # Hairpins
    h_count = 0
    if edges_by_type.get("H"):
        # Extract hairpins
        for h_edges in edges_by_type["H"]:
            h_count += 1
            h_start, h_stop = h_edges
            h_seq = seq[h_start - 1:h_stop]  # -1 for zero-based

            pos5 = h_start - 1  # -1 for flanking
            pos3 = h_stop + 1  # +1 for flanking

            nuc1 = seq[pos5 - 1:pos5]  # subtrack 1 to get zero-based to get nucleotides flanking (closing base pair)
            nuc2 = seq[pos3 - 1:pos3]

            regions["H"].append([h_start, h_stop])

            hknots = includes_knot(h_start, h_stop, knots)
            pk = hknots if hknots else ""

            structure_types["H"].append(f"H{h_count} {h_start}..{h_stop} \"{h_seq}\" ({pos5},{pos3}) {nuc1}:{nuc2} {pk}\n")

            for k in hknots:
                pk_loops[k].append([f"H{h_count}", h_start, h_stop])

    ###################
    # Extract regions #
    ###################
    prev_char = ""
    this_start = 0
    this_stop = 0
    FIRST = 1
    ####################################################
    # Stems, internal loops, multiloops, hairpin loops #
    ####################################################
    for i in range(len(s)):
        if not s[i]:
            raise Exception(f"No value in s for {i}")

        if s[i] != prev_char:
            # In a new region
            # Store previous region info:
            # Ignore multiloops and hairpins, they are already added above
            if prev_char not in ["M", "H"]:
                # Store in 1-based to be consistent with M and H above
                if prev_char not in regions: regions[prev_char] = []
                if not FIRST: regions[prev_char].append([this_start + 1, this_stop + 1])

            # Reset the start/stop:
            this_start = i
            this_stop = i
            prev_char = s[i]
            FIRST = 0

        elif (prev_char == "S") and (bp[i + 1] + 1 != bp[i]):
            # In a break in a stem into another stem. Store in 1-based
            regions[prev_char].append([this_start + 1, this_stop + 1])
            this_start = i
            this_stop = i
            prev_char = s[i]
            FIRST = 0

        elif (prev_char == "S") and (bp[i] == i + 1):
            # In a break in a stem into another stem. Store in 1-based
            regions[prev_char].append([this_start + 1, this_stop + 1])
            this_start = i
            this_stop = i
            prev_char = s[i]
            FIRST = 0

        else:
            # Continue/extend the same region
            this_stop = i

    # Store the last region
    if prev_char not in ["M", "H"]:
        regions[prev_char].append([this_start + 1, this_stop + 1])

    # Bulges
    if regions.get("B"):
        b_count = 0
        for bulge in regions["B"]:
            # 1-based
            b_start, b_stop = bulge

            # all substr calls need 0-based, hence -1 for all:
            b_seq = seq[b_start - 1:b_stop - 1]  # -1 for zero-based

            # 1-based positions of flanking:
            bp5_pos1 = b_start - 1
            bp3_pos1 = b_stop + 1

            bp5_nt1 = seq[bp5_pos1 - 1:bp5_pos1]
            bp3_nt1 = seq[bp3_pos1 - 1:bp3_pos1]

            bp5_pos2 = bp[bp5_pos1]
            bp3_pos2 = bp[bp3_pos1]

            bp5_nt2 = seq[bp5_pos2 - 1:bp5_pos2]
            bp3_nt2 = seq[bp3_pos2 - 1:bp3_pos2]

            b_count += 1
            bknots = includes_knot(b_start, b_stop, knots)
            pk = bknots if bknots else ""

            structure_types["B"].append(
                f"B{b_count} {b_start}..{b_stop} \"{b_seq}\" ({bp5_pos1},{bp5_pos2}) {bp5_nt1}:{bp5_nt2} ({bp3_pos1},{bp3_pos2}) {bp3_nt1}:{bp3_nt2} {pk}\n"
            )

            for k in bknots:
                pk_loops[k].append([f"B{b_count}", b_start, b_stop])
    
    if regions.get("I"):
        iG = nx.Graph()

        for i in range(len(regions["I"])):
            i_start, i_stop = regions["I"][i]

            for j in range(i + 1, len(regions["I"])):
                j_start, j_stop = regions["I"][j]

                if loop_linked(i_start, i_stop, j_start, j_stop, bp):
                    iG.add_edge(i, j)

        i_components = nx.connected_components(iG)

        internal_loops = []
        for c in i_components:
            sorted_c = sorted(
                c, key=lambda x: regions["I"][x][0]
            )  # Sort by the starting position of each internal loop
            internal_loops.append(sorted_c)
    i_count = 0

    # Sort internal loops by their starting positions
    internal_loops = sorted(
        internal_loops, key=lambda x: regions["I"][x[0]][0]
    )  # Sort by the starting position of each internal loop

    for c in internal_loops:
        i_count += 1

        component_size = len(c)
        ip = 0

        for v in c:
            ip += 1

            # 1-based positions
            i_start, i_stop = regions["I"][v]
            i_length = i_stop - i_start + 1

            # Subtract 1 to convert to 0-based
            i_seq = seq[i_start - 1:i_start + i_length - 1]

            # Positions in 1-based here:
            bp5_pos1 = i_start - 1
            bp5_pos2 = bp[bp5_pos1]

            nuc5_1 = seq[bp5_pos1 - 1:bp5_pos1]
            nuc5_2 = seq[bp5_pos2 - 1:bp5_pos2]

            iknots = includes_knot(i_start, i_stop, knots)
            pk = iknots if iknots else ""

            structure_types["I"].setdefault(i_count, []).append(
                f"I{i_count}.{ip} {i_start}..{i_stop} \"{i_seq}\" ({bp5_pos1},{bp5_pos2}) {nuc5_1}:{nuc5_2} {pk}\n"
            )

            for k in iknots:
                pk_loops[k].append([f"I{i_count}.{ip}", i_start, i_stop])
    
    if regions["E"]:
        e_count = 0
        for e in regions["E"]:
            e_start, e_stop = e
            e_seq = seq[e_start - 1:e_stop]

            e_count += 1
            eknots = includes_knot(e_start, e_stop, knots)
            pk = eknots if eknots else ""

            structure_types["E"].append(
                f"E{e_count} {e_start}..{e_stop} \"{e_seq}\" {pk}\n"
            )

            for k in eknots:
                pk_loops[k].append([f"E{e_count}", e_start, e_stop])

    # Stems
    visited = {}  # A hash to keep track of regions in stems that have been collected already
    stem_list = []
    if regions["S"]:
        s_count = 0
        for stem in sorted(regions["S"], key=lambda x: x[0]):
            s_start1, s_stop1 = stem

            if not visited.get(s_start1, {}).get(s_stop1):
                s_count += 1

                # 1-based
                s_start2 = bp[s_stop1]
                s_stop2 = bp[s_start1]

                # All substr calls need 0-based, hence -1 for all:
                s_seq1 = seq[s_start1 - 1:s_stop1]
                s_seq2 = seq[s_start2 - 1:s_stop2]

                structure_types["S"].append(
                    f"S{s_count} {s_start1}..{s_stop1} \"{s_seq1}\" {s_start2}..{s_stop2} \"{s_seq2}\"\n"
                )

                if s_start2 not in visited: visited[s_start2] = {}
                visited[s_start2][s_stop2] = 1
                stem_list.append([s_start1, s_stop1, f"S{s_count}"])

    nc_count = 0

    for i, j in bp.items():
        if i < j:
            b1 = seq[i - 1]
            b2 = seq[j - 1]

            if non_canonical(b1, b2):
                nc_count += 1
                this_label = ""

                for stem in stem_list:
                    start, stop, label = stem

                    if start <= i <= stop:
                        this_label = label

                if not this_label:
                    raise Exception(f"No label found for {i}, {j}\n")

                structure_types["NCBP"].append(
                    f"NCBP{nc_count} {i} {b1} {j} {b2} {this_label}\n"
                )

        if knots:
            for k in range(len(knots)):
                knot_id = k + 1
                knot = knots[k]
                knot_size = len(knot)

                first = knot.pop(0)
                k_5p_start, k_3p_start = first

                last = knot.pop() if knot else first
                k_3p_stop, k_5p_stop = last

                linked_loops = ""

                if len(pk_loops[knot_id]) == 2:
                    pk_loops[knot_id] = sorted(pk_loops[knot_id], key=lambda x: x[1])
                    l_type1, l_start1, l_stop1 = pk_loops[knot_id][0]
                    l_type2, l_start2, l_stop2 = pk_loops[knot_id][1]

                    linked_loops = f"{l_type1} {l_start1}..{l_stop1} {l_type2} {l_start2}..{l_stop2}"
                else:
                    raise Exception(f"Expected two loops linked for PK{knot_id}\n")

                structure_types["PK"].append(
                    f"PK{knot_id} {knot_size}bp {k_5p_start}..{k_3p_stop} {k_5p_stop}..{k_3p_start} {linked_loops}\n"
                )

                stem_list.append([k_5p_start, k_3p_stop, f"PK{knot_id}"])

                n = 0
                for pair in knot:
                    k_5p, k_3p = pair

                    # Positions in 1-based
                    b_5p = seq[k_5p - 1]
                    b_3p = seq[k_3p - 1]

                    n += 1
                    structure_types["PKBP"].append(f"PK{knot_id}.{n} {k_5p} {b_5p} {k_3p} {b_3p}\n")

                    # Check if PKBP is non-canonical.
                    if non_canonical(b_5p, b_3p):
                        nc_count += 1
                        structure_types["NCBP"].append(
                            f"NCBP{nc_count} {k_5p} {b_5p} {k_3p} {b_3p} PK{knot_id}.{n}\n"
                        )
    for i, segment in enumerate(segments):
        segment_id = i + 1
        seg_size = len(segment)

        first_pair = segment.pop(0)
        seg_5p_start, seg_3p_start = first_pair

        last_pair = segment.pop() if segment else first_pair
        seg_3p_stop, seg_5p_stop = last_pair

        seg_seq1 = seq[seg_5p_start - 1:seg_3p_stop]
        seg_seq2 = seq[seg_5p_stop - 1:seg_3p_start]

        structure_types["SEGMENTS"].append(
            f"segment{segment_id} {seg_size}bp {seg_5p_start}..{seg_3p_stop} {seg_seq1} {seg_5p_stop}..{seg_3p_start} {seg_seq2}\n"
        )

    k = list("N" * len(dotbracket))

    for i, pk_value in enumerate(s):
        if pk_value:
            k[i] = "K"

    return dotbracket, s, k, structure_types, page_number


def build_segment_graph(seq, bp, segments, knots):
    first_pos, last_pos = get_extreme_positions(bp)
    G = nx.Graph()
    edges = []

    for knot in knots:
        first_pair = knot[0]
        k1_5p_start, k1_3p_start = first_pair
        last_pair = knot[-1] if len(knot) > 1 else first_pair
        k1_3p_stop, k1_5p_stop = last_pair

        k1_seq1 = seq[k1_5p_start - 1:k1_3p_stop]
        k1_seq2 = seq[k1_5p_stop - 1:k1_3p_start]

    for i, segment in enumerate(segments):
        if not segment: continue
        G.add_node(i)
        first_pair = segment[0]
        s1_5p_start, s1_3p_start = first_pair
        last_pair = segment[-1] if len(segment) > 1 else first_pair
        s1_3p_stop, s1_5p_stop = last_pair

        s1_seq1 = seq[s1_5p_start - 1:s1_3p_stop]
        s1_seq2 = seq[s1_5p_stop - 1:s1_3p_start]

        p1_5p_start = get_prev_pair(s1_5p_start, bp, first_pos, knots)
        n1_3p_stop = get_next_pair(s1_3p_stop, bp, last_pos, knots)
        p1_5p_stop = get_prev_pair(s1_5p_stop, bp, first_pos, knots)
        n1_3p_start = get_next_pair(s1_3p_start, bp, last_pos, knots)

        for j, other_segment in enumerate(segments):
            if not other_segment: continue
            s2_5p_start, s2_3p_start = other_segment[0]
            s2_3p_stop, s2_5p_stop = other_segment[-1] if len(other_segment) > 1 else first_pair

            if n1_3p_start == s2_5p_start:
                G.add_edge(i, j)
                edges.append((i, j, s1_3p_start, s2_5p_start, "1"))

            if n1_3p_start == s2_5p_stop:
                G.add_edge(i, j)
                edges.append((i, j, s1_3p_start, s2_5p_stop, "2"))

            if n1_3p_stop == s2_5p_start:
                G.add_edge(i, j)
                edges.append((i, j, s1_3p_stop, s2_5p_start, "3"))

            if n1_3p_stop == s2_5p_stop:
                G.add_edge(i, j)
                edges.append((i, j, s1_3p_stop, s2_5p_stop, "4"))

    return G, edges


def compute_structure_array(dotbracket, bp, seq):
    knot_bracket = get_knot_brackets()
    structure_array = []
    pseudoknot_array = []

    for i, x in enumerate(dotbracket):
        loop_structure = ""

        # stem
        if x == "(" or x == ")":
            loop_structure = "S"
        elif (x == "." or x in knot_bracket):
            # loops
            fwd, fwd_index = fwd_finder(i, dotbracket, x, knot_bracket)
            # bwd, bwd_index = bwd_finder(i, dotbracket, x, knot_bracket)
            bwd, bwd_index = bwd_finder(len(dotbracket) - i-1, dotbracket, x, knot_bracket)

            fwd_index_pair = bp[fwd_index + 1]  # returns position on RNA (1-based)
            bwd_index_pair = bp[bwd_index + 1] if bwd_index != -1 else fwd_index_pair
            length = fwd_index - bwd_index - 1

            if bwd == "(":
                if fwd == "(":
                    if fwd_index_pair == bwd_index_pair - 1:
                        loop_structure = "B"
                    else:
                        loop_structure = between(fwd_index_pair - 1, bwd_index_pair - 1, dotbracket)
                elif fwd == ")":
                    loop_structure = "H"
                elif fwd == "":
                    loop_structure = "E"
                else:
                    raise ValueError(f"Unrecognized forward base: {fwd}")

            elif bwd == ")":
                if fwd == "(":
                    loop_structure = "X"
                elif fwd == ")":
                    if fwd_index_pair == bwd_index_pair - 1:
                        loop_structure = "B"
                    else:
                        loop_structure = between(fwd_index_pair - 1, bwd_index_pair - 1, dotbracket)
                elif fwd == "":
                    loop_structure = "E"
                else:
                    raise ValueError(f"Unrecognized forward base: {fwd}")

            elif bwd == "" or bwd == ".":
                loop_structure = "E"
            else:
                raise ValueError(f"Unrecognized backward base: {bwd}")

        structure_array.append(loop_structure)

    pseudoknot_array = [0] * len(dotbracket)
    for i in range(len(dotbracket)):
        x = dotbracket[i]
        if x in knot_bracket:
            pseudoknot_array[i] = 1

    return structure_array, pseudoknot_array


def print_structure_data(regions):
    # collect lines for output file
    lines = []
    for type in regions:
        if type and type != "N":
            count = 0
            for region in regions[type]:
                count += 1
                start, stop = region
                lines.append([start, f"{id}\tbpRNA\t{type}\t{start}\t{stop}\t.\t+\t.\tID={type}{count}"])
    
    gff_file = f"{id}_structure.gff"
    with open(gff_file, "w") as gff:
        for region in sorted(lines, key=lambda x: x[0]):
            start, line = region
            gff.write(f"{line}\n")


def is_multiloop(og_components, mG, edges):
    
    if len(og_components) == 1:
        return 0
    # make copy of components
    components = list(og_components)

    first_vertex = components.pop(0)
    current_vertex = first_vertex
    while components:
        successors = list(mG.successors(current_vertex))
        if successors:
            # It should only have one successor. Reality check here.
            if len(successors) != 1:
                raise Exception("Fatal error: Too many successors of multiloop part.")
            next_vertex = successors[0]
            index = -1
            for i, component in enumerate(components):
                if component == next_vertex:
                    index = i
            if index >= 0:
                components.pop(index)
            else:
                return 0
            current_vertex = next_vertex

        else:
            return 0

    # If here, all vertices in @c were removed.
    # Check if successor of last node is first node.
    return mG.has_edge(current_vertex, first_vertex)


def compute_dot_bracket(segments, knots, seq):
    dotbracket = np.array(list("." * len(seq)))

    # Set all pairs to parentheses for the segments
    for segment in segments:
        for pair in segment:
            l, r = pair
            dotbracket[l - 1] = "("
            dotbracket[r - 1] = ")"

    # Loop through each base pair and update bracket
    page = {}
    n = 1
    unlabeled_knots = 1
    first_i = 0

    while unlabeled_knots:
        # Assume no more left
        unlabeled_knots = 0

        for i in range(len(knots)):
            if i not in page:
                first_i = i
                break

        page[first_i] = n

        # The most 5' knot with undefined page is what everything is compared to
        # Start an array containing members of this page
        check_list = [first_i]

        # Start at next knot with undefined page
        for i in range(first_i + 1, len(knots)):
            if i not in page:
                # For now, consider it not labeled
                crossing = 0

                # If it doesn't cross, then use $n, otherwise label later.
                for check_i in check_list:
                    if knots_cross(knots[check_i], knots[i]):
                        # There is an unlabeled knot left.
                        unlabeled_knots = 1
                        crossing = 1
                        break

                if not crossing:
                    page[i] = n
                    check_list.append(i)

        n += 1

    page_number = 0
    for i in range(len(knots)):
        n = page[i]  # page of PK. For all PKs, n > 0, where n=0 is "("

        if n > page_number:
            page_number = n

        for pair in knots[i]:
            l, r = pair
            lB, rB = get_brackets(n)
            dotbracket[l - 1] = lB
            dotbracket[r - 1] = rB

    page_number += 1  # Convert to 1-based
    return ''.join(dotbracket), page_number


def knotsOverlap(knot1, knot2):
    # treat $knot like a segment
    #
    # 5'start   3'stop
    #       ACGUA
    #       |||||
    #       UGCAU
    # 3'start   5'stop
    #   
    k1_5pStart, k1_3pStart = knot1[0]
    k2_5pStart, k2_3pStart = knot2[0]
    k1_3pStop, k1_5pStop = knot1[-1]
    k2_3pStop, k2_5pStop = knot2[-1]

    if (k1_5pStart <= k2_5pStart <= k1_3pStart) or \
       (k1_5pStart <= k2_3pStart <= k1_3pStart):
        return 1

    return 0


def knots_cross(knot1, knot2):
    # treat $knot like a segment
    #
    # 5'start   3'stop
    #       ACGUA
    #       |||||
    #       UGCAU
    # 3'start   5'stop
    #   
    k1_5pStart, k1_3pStart = knot1[0]
    k2_5pStart, k2_3pStart = knot2[0]
    if pkQuartet(k1_5pStart, k1_3pStart, k2_5pStart, k2_3pStart):
        return 1
    return 0

def loop_linked(iStart1, iStop1, iStart2, iStop2, bp):
    # 0123456789X
    # ((.))((.))...
    # here it is forced to be in order 5' to 3'
    if iStop1 + 1 == bp[iStart2 - 1]:
        return 1
    return 0


def get_knot_brackets():
    knots = "[]{}<>" + "".join([chr(i) for i in range(ord('a'), ord('z') + 1)]) + "".join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    knotBracket = {}
    for c in knots:
        knotBracket[c] = 1
    return knotBracket


def get_brackets(n):
    if n >= 30:
        raise Exception("Fatal error: too many (n>29) PKs to represent in dotbracket!")
    left = "([{<" + "".join([chr(i) for i in range(ord('A'), ord('Z') + 1)])
    right = ")]}>" + "".join([chr(i) for i in range(ord('a'), ord('z') + 1)])
    return left[n], right[n]


# Find index of the next paired base
def fwd_finder(i, dotbracket, x, knotBracket):
    B = dotbracket[i]
    while ((B == ".") or (B in knotBracket)) and (i+1 < len(dotbracket)):
        i += 1
        B = dotbracket[i]
        if i >= len(dotbracket):
            return ("", i)
    return (B, i)


# Find index of the previous paired base
def bwd_finder(i, dotbracket, x, knotBracket):
    B = dotbracket[i]
    while ((B == ".") or (B in knotBracket)) and (i < len(dotbracket)-1):
        i -= 1
        B = dotbracket[i]
        if i < 0:
            return("", i)
    return(B, i)


# Determine if multiloop or internal loop
# Usage:			# get to zero base
# $type = between($fwdPairIndex-1,$bwdPairIndex-1,$dotbracket);
def between(fwdIP, bwdIP, dotbracket):
    fwdIP += 1
    while fwdIP < bwdIP - 1:
        fwdP = dotbracket[fwdIP]
        if fwdP == "(" or fwdP == ")":
            return "X"
        fwdIP += 1
    return "I"


def pkQuartet(i, j, k, l):
    # Assumption: i < j and k < l
    if ((i < k and k < j and j < l) or
        (i < l and l < j and k < i)):
        return 1
    return 0


def get_segments(bp):
    # 5'start    3'stop
    #       ACG.UA
    #       ||| ||
    #       UGCAAU
    # 3'start    5'stop
    #

    # Initialize an empty list to store all segments
    all_segments = []

    # Get the first and last positions of the base pairs
    first_pos, last_pos = get_extreme_positions(bp)

    # Initialize the current position indices
    i = get_next_pair(0, bp, last_pos, knots=[])
    if i:
        # Build the first segment starting with the first base pair
        in_segment = False
        while first_pos <= i <= last_pos:
            j = bp[i]
            segment = []
            if i < j: 
                segment.append([i, j])
                in_segment = True

            # Grow the segment to include more base pairs
            while in_segment:
                next_pair = get_next_pair(i, bp, last_pos, knots=[])
                prev_pair = get_prev_pair(j, bp, first_pos, knots=[])

                if next_pair and prev_pair:
                    if (bp[next_pair] == prev_pair) and (next_pair < prev_pair):
                        # Add the base pair to the segment
                        segment.append([next_pair, prev_pair])
                        i = next_pair
                        j = prev_pair
                    else:
                        # Close the segment and add it to the list
                        in_segment = False
                        # if segment: 
                        all_segments.append(segment)
                else:
                    in_segment = False
                    # if segment: 
                    all_segments.append(segment)

            # Move to the first base pair of the next segment
            i = get_next_pair(i, bp, last_pos, knots=[])

    # By construction, the segments should be ordered by the most 5' position of each segment
    return all_segments


def filter_base_pairs(bp, knots):
    filtered_bp = {}
    for i, j in bp.items():
        if not in_knot(i, knots):
            if not in_knot(j, knots):
                filtered_bp[i] = j
            else:
                raise ValueError(f"filterBasePairs: {j} in PK, but {i} is not.")

    return filtered_bp


def separate_segments(segments):
    warnings = ""
    knot = {}
    G = nx.Graph()

    if len(segments) > 1:
        for i in range(len(segments)-1):
            segment1 = segments[i]
            if not segment1: continue
            firstPair1 = segment1[0]
            s1_5pStart, s1_3pStart = firstPair1

            for j in range(i + 1, len(segments)):
                segment2 = segments[j]
                if not segment2: continue
                firstPair2 = segment2[0]
                s2_5pStart, s2_3pStart = firstPair2

                if pkQuartet(s1_5pStart, s1_3pStart, s2_5pStart, s2_3pStart):
                    G.add_edge(i, j)

    CCs = list(nx.connected_components(G))
    ccCount = 0
    for c in CCs:
        if DEBUG: print("Primary Connected Component: ", G.subgraph(c))
        for v in c:
            if DEBUG: print("w($v)=", len(segments[v]))
        ccCount += 1
        knotsList, warning = get_best_knots(G, c, segments)
        for v in knotsList:
            if G.has_node(v):
                G.remove_node(v)
            knot.setdefault(v, 0)
            knot[v] += 1
        warnings += warning

    if segments:
        if len(segments) == len(knot):
            maxSize = 0
            maxI = None
            for i in range(len(segments)):
                if len(segments[i]) > maxSize:
                    maxI = i
                    maxSize = len(segments[i])

            del knot[maxI]
            
    knots = []
    all_segments = []
    for i in range(len(segments)):
        if knot.get(i):
            knots.append(segments[i])
        else:
            all_segments.append(segments[i])

    return all_segments, knots, warnings


def get_min_v_pair(c, segments):
    v = c[0]
    w = c[1]

    # Check the PK sequence
    if len(segments[v]) < len(segments[w]):
        return (v, "")
    elif len(segments[w]) < len(segments[v]):
        return (w, "")
    else:
        Nv = len(segments[v])
        Nw = len(segments[w])
        print(f"Checking {v} and {w} (CC of size 2), w({v})={Nv} and w({w})={Nw}")

        # Debug information
        print(f"{v}:")
        for pair in segments[v]:
            f, t = pair
            print(f"{v}: {f},{t}")

        print(f"{w}:")
        for pair in segments[w]:
            f, t = pair
            print(f"{w}: {f},{t}")

        minV = min(v, w)

        warning = "#Warning: Structure contains linked PK-segments of same sizes {} and {}. Using PK brackets for the more 5' segment\n".format(Nv, Nw)

        return (minV, warning)
    
    
def getBestKnot(G, component, segments):
    # find the knot with the most segments
    maxSegments = 0
    bestKnot = None
    for node in component:
        nodeSegments = len(set(segments[node]))
        if nodeSegments > maxSegments:
            maxSegments = nodeSegments
            bestKnot = node

    return bestKnot


def get_best_knots(G, c, segments, DEBUG=False):
    warnings = ""
    knots_list = []

    # Create a subgraph of the connected component
    g = G.subgraph(c)
    g = nx.Graph(g)

    # Check for connected components
    connected_components = list(nx.connected_components(g))

    # Initialize variables
    knots_remain = False
    min_v = None
    min_weight = 10e10
    max_degree = 0
    max_max_degree_score = -1
    max_degree_v = None
    node_info = {}

    # Iterate over connected components
    for cc in connected_components:
        if len(cc) == 2:
            knots_remain = True
            (min_v, warning) = get_min_v_pair(cc, segments)
            warnings += warning
            g.remove_node(min_v)
            knots_list.append(min_v)
            if DEBUG:
                print(f"2-deleted {min_v}")

        elif len(cc) > 2:
            if is_path_graph(g, cc):
                knots_remain = True

                # Get the path and its corresponding weights
                path = list(is_path_graph(g, cc))
                path_weights = [len(segments[v]) for v in path]

                # Check for a specific path pattern
                if len(path) == 3 and path_weights[1] == path_weights[0] + path_weights[2]:
                    v = path[1]
                    g.remove_node(v)
                    knots_list.append(v)

                    if DEBUG:
                        print(f"3-path-deleted {v}")

                else:
                    # Initialize variables for finding maximum weighted vertices
                    max_set = [0 for _ in range(len(path) + 1)]
                    max_set[1] = len(segments[path[0]])

                    for i in range(2, len(path) + 1):
                        weight1 = len(segments[path[i-2]])
                        weight2 = len(segments[path[i-1]])
                        max_set[i] = max(max_set[i-1], max_set[i-2] + weight2)

                    max_weighted = {}

                    # Identify maximum weighted vertices
                    i = len(path)
                    while i >= 1:
                        weight1 = len(segments[path[i-2]])
                        weight2 = len(segments[path[i-1]])

                        if i == 2:
                            if weight1 == weight2:
                                max_v = max(path[i-2], path[i-1])
                                max_weighted[max_v] = 1
                                i -= 1
                                break

                        elif i == len(path):
                            if weight1 == weight2:  # if w(i-1) == w(i)
                                if max_set[i] == max_set[i-1]:
                                    if path[i-1] > path[i-2]:
                                        max_weighted[path[i-1]] = 1
                                        i -= 2
                                        continue

                        if max_set[i] == max_set[i-1]:
                            i -= 1
                        else:
                            max_weighted[path[i-1]] = 1
                            i -= 2

                    # Delete vertices that are not maximum weighted
                    for v in path:
                        if v not in max_weighted:
                            knots_remain = True
                            g.remove_node(v)
                            knots_list.append(v)

                            if DEBUG:
                                print(f"path-deleted {v}")

            else:  # Complex graph with more than 2 nodes
                knots_remain = True

                # Initialize variables
                min_v = ""
                min_weight = float("inf")
                max_degree = 0
                max_max_degree_score = -1
                max_degree_v = None
                node_info = {}

                # Iterate through all vertices of this connected component
                for v in cc:
                    # Get the degree and weight for the current node
                    d = G.degree(v)
                    weight = len(segments[v])

                    # Update the minimum weight node
                    if weight < min_weight:
                        min_weight = weight
                        min_v = v

                    # Update the highest degree and score
                    if d >= max_degree:
                        if d > max_degree:
                            max_max_degree_score = -1

                        # Compute the sum of the neighbor's weights
                        weight_sum = 0
                        for w in G.neighbors(v):
                            weight_sum += len(segments[w])

                        if DEBUG: print("weight sum:", weight_sum, "\n") 
                        if DEBUG: print("v=", v, ", weight:", len(segments[v]), "\n") 

                        score = weight_sum - weight
                        node_info[v] = [d, score, weight]

                        if score > max_max_degree_score:
                            max_degree_v = v
                            max_degree = d
                            max_max_degree_score = score

                # Count nodes with maximum degree and max score
                count = 0
                for v in node_info:
                    d, score, weight = node_info[v]
                    if (d == max_degree) and (score == max_max_degree_score):
                        count += 1

                if DEBUG: print("two nodes:", id, "\n") 

                # Delete the node with maximum degree and max score if it exists
                if max_max_degree_score > 0:
                    G.remove_node(max_degree_v)
                    knots_list.append(max_degree_v)
                    if DEBUG: print("degree-deleted", max_degree_v, "\n") 

                    # Iterate through segments of the deleted node
                    for pair in segments[max_degree_v]:
                        f, t = pair
                        if DEBUG: print(max_degree_v, ":", f, ",", t, "\n") 

                # Otherwise, delete the minimum weight node
                else:
                    G.remove_node(min_v)
                    knots_list.append(min_v)
                    if DEBUG: print("minweight-deleted", min_v, "\n") 

    if DEBUG: print(f"All knots eliminted for this component: {knots_list} found.")
    return knots_list, warnings


def getBestKnots(G, c, segments):
    warnings = ""

    # Initialize variables
    knots_list = []
    g = G.subgraph(c)
    knots_remain = True

    # Iterate until no more knots remain
    while knots_remain:
        if DEBUG: print("getBestKnots: Current graph:", g, "\n") 

        # Recompute connected components
        connected_components = list(nx.connected_components(g))

        # Process each connected component
        for cc in connected_components:
            # Check for 2-node component
            if len(cc) == 2:
                if DEBUG: print("2-checking:", cc, "\n") 

                # Find the minimum vertex pair
                min_v, warning = get_min_v_pair(g, cc, segments)
                warnings += warning

                # Remove the minimum vertex
                g.remove_node(min_v)
                knots_list.append(min_v)
                if DEBUG: print("2-deleted:", min_v, "\n") 

                knots_remain = True
                break

            # Check for path graph with 3 nodes
            elif len(cc) == 3 and is_path_graph(g, cc):
                if DEBUG: print("found a path:", cc, "\n") 

                # Get the path
                path = list(is_path_graph(g, cc))

                # Check for specific weight condition
                w1 = len(segments[path[0]])
                w2 = len(segments[path[1]])
                w3 = len(segments[path[2]])

                if w2 == w1 + w3:
                    if DEBUG: print("3-path-checking:", path, "\n") 

                    # Remove the middle vertex
                    v = path[1]
                    g.remove_node(v)
                    knots_list.append(v)
                    if DEBUG: print("3-path-deleted:", v, "\n") 

                    knots_remain = True
                    break

                # Initialize variables for maximum weight set
                max_set = [0, len(segments[path[0]])]

                # Find maximum weight set for the path
                for i in range(2, len(path)):
                    # the path index is 0-based, but maxSet is 1-based
                    # 0th term is 0.
                    w = len(segments[path[i]])
                    max_set.append(max(max_set[i - 1], max_set[i - 2] + w))

                # Check if the maximum weight set is the entire path
                if max_set[-1] == sum(len(segments[p]) for p in path):
                    if DEBUG: print("3-path-maxset:", path, "\n") 

                    # Remove the first vertex
                    v = path[0]
                    g.remove_node(v)
                    knots_list.append(v)
                    if DEBUG: print("3-path-maxset-deleted:", v, "\n") 

                    knots_remain = True
                    break

            # Check for other cases (not implemented)
            else:
                if DEBUG: print("Unhandled component:", cc, "\n") 
                
            if DEBUG: print(max_set, "length=", len(max_set), "\n") 

            i = len(path)
            max_weighted = {}

            # Only apply path algorithm to all but last two members
            while i >= 1:
                weight1 = len(segments[path[i - 2]])
                weight2 = len(segments[path[i - 1]])

                if DEBUG: print("checking", i, ":", max_set[i - 2] + weight2, "vs", max_set[i - 1]) 

                if i == 2:
                    if weight1 == weight2:
                        max_v = max(path[i - 2], path[i - 1])
                        max_weighted[max_v] = 1
                        i -= 1
                        break

                if i == len(path):
                    if weight1 == weight2:  # If w(i-1) == w(i)
                        if max_set[i] == max_set[i - 1]:
                            if path[i - 1] > path[i - 2]:
                                max_weighted[path[i - 1]] = 1
                                i -= 2
                                continue

                if max_set[i-1] == max_set[i - 2]:
                    i -= 1
                else:
                    if DEBUG: print("storing", i, ":", path[i - 1])
                    max_weighted[path[i - 1]] = 1
                    i -= 2

            # Remove unweighted nodes from the path
            if nx.is_frozen(g): g = nx.Graph(g)
            for v in path:
                if v not in max_weighted:
                    knots_remain = True
                    g.remove_node(v)
                    knots_list.append(v)
                    if DEBUG: print("path-deleted", v)

            for i in range(len(path)):
                v = path[i]

                if not max_weighted.get(v) and (v in g):
                    g.remove_node(v)
                    knots_list.append(v)
                    if DEBUG: print("path-deleted", v, "\n") 
                    knots_remain = True
                    break
                else:
                    # complex graph with more than 2 nodes
                    knots_remain = True

                    # Initialize variables
                    min_v = ""
                    min_weight = float("inf")
                    max_degree = 0
                    max_max_degree_score = -1
                    max_degree_v = None
                    node_info = {}

                    # Iterate through all vertices of this connected component
                    for v in cc:
                        # Get the degree and weight for the current node
                        if not G.has_node(v): 
                            raise ValueError(f'Graph should have node {v}')
                        d = G.degree(v)
                    
                        weight = len(segments[v])

                        # Update the minimum weight node
                        if weight < min_weight:
                            min_weight = weight
                            min_v = v

                        # Update the highest degree and score
                        if d >= max_degree:
                            if d > max_degree:
                                max_max_degree_score = -1

                            # Compute the sum of the neighbor's weights
                            weight_sum = 0
                            for w in G.neighbors(v):
                                weight_sum += len(segments[w])

                            if DEBUG: print("weight sum:", weight_sum, "\n") 
                            if DEBUG: print("v=", v, ", weight:", len(segments[v]), "\n") 

                            score = weight_sum - weight
                            node_info[v] = [d, score, weight]

                            if score > max_max_degree_score:
                                max_degree_v = v
                                max_degree = d
                                max_max_degree_score = score

                    # Count nodes with maximum degree and max score
                    count = 0
                    for v in node_info:
                        d, score, weight = node_info[v]
                        if (d == max_degree) and (score == max_max_degree_score):
                            count += 1

                    if DEBUG: print("two nodes:", id, "\n") 

                    # Delete the node with maximum degree and max score if it exists
                    if max_max_degree_score > 0:
                        G.remove_node(max_degree_v)
                        knots_list.append(max_degree_v)
                        if DEBUG: print("degree-deleted", max_degree_v, "\n") 

                        # Iterate through segments of the deleted node
                        for pair in segments[max_degree_v]:
                            f, t = pair
                            if DEBUG: print(max_degree_v, ":", f, ",", t, "\n") 

                    # Otherwise, delete the minimum weight node
                    else:
                        G.remove_node(min_v)
                        knots_list.append(min_v)
                        if DEBUG: print("minweight-deleted", min_v, "\n") 

    if DEBUG: print(f"All knots eliminted for this component: {knots_list} found.")
    return knots_list, warnings


def is_path_graph(G, c):
    # first find two nodes with 1 edge
    ends = []
    if len(c) == 1:
        return False

    for v in c:
        degree = G.degree(v)
        if degree == 1:
            ends.append(v)

    if len(ends) > 2:
        return False  # more than 2 1-degree vertices means it is not a path graph

    if len(ends) < 2:
        return False  # can't be a path graph
    else:
        start, end = ends
        path = [start]

        while len(path) < len(c):
            neighbors = list(G.neighbors(path[-1]))
            if len(neighbors) == 2:
                a, b = neighbors
                if a == path[-2]:
                    path.append(b)
                elif b == path[-2]:
                    path.append(a)
                else:
                    raise ValueError(f"Unexpected neighbor: {neighbors}")
            elif len(neighbors) == 1:
                if path[0] == start:
                    path.append(neighbors[0])
                elif neighbors[0] == end:
                    path.append(neighbors[0])
                else:
                    raise ValueError(f"Unexpected break: {neighbors}")
            else:
                return False
        return path


def min(x, y):
    return x if x < y else y

def max(a, b):
    return b if b > a else a

def get_extreme_positions(bp):
    if bp:
        keys = list(bp.keys())
        length = len(keys)
        sortedKeys = sorted(keys)
        firstKey = sortedKeys[0]
        lastKey = sortedKeys[-1]
        return firstKey, lastKey
    else:
        raise ValueError(f"No basepairs found for {id}")


def get_next_pair(i, bp, last_pos, knots):
    for n in range(i + 1, last_pos + 1):
        if not in_knot(n, knots):
            if n in bp:
                return n

    return 0


def get_prev_pair(j, bp, first_pos, knots):
    for p in range(j - 1, first_pos - 1, -1):
        if not in_knot(p, knots):
            if p in bp:
                return p

    return 0


def non_canonical(base1, base2):
    canonical_pairs = {
        ("A", "U"),
        ("C", "G"),
        ("G", "U"),
    }

    if (base1, base2) in canonical_pairs or (base2, base1) in canonical_pairs:
        return 0
    else:
        return 1


def in_knot(pos, knots):

    for knot in knots:
        # Treat each knot as a segment
        #
        # 5'start   3'stop
        #       ACGUA
        #       |||||
        #       UGCAU
        # 3'start   5'stop
        #
        k_5p_start, k_3p_start = knot[0]
        k_3p_stop, k_5p_stop = knot[-1]

        if (k_5p_start <= pos <= k_3p_stop) or (k_5p_stop <= pos <= k_3p_start):
            return True

    return False


def includes_knot(start, stop, knots):
    loop_knots = []

    if knots:
        for k, knot in enumerate(knots):
            # Treat each knot as a segment
            #
            # 5'start   3'stop
            #       ACGUA
            #       |||||
            #       UGCAU
            # 3'start   5'stop
            #
            k_5p_start, k_3p_start = knot[0]
            k_3p_stop, k_5p_stop = knot[-1]

            if start <= k_5p_start and k_3p_stop <= stop:
                loop_knots.append(k + 1)  # Node label is 1-based

            if start <= k_5p_stop and k_3p_start <= stop:
                loop_knots.append(k + 1)  # Node label is 1-based

    return loop_knots


def pair_map(dotbracket):
    stack = {}
    pmap = [0] * len(dotbracket)

    for i, char in enumerate(dotbracket):
        if char in "([{<A-Z]":
            stack.setdefault(char, []).append(i)
        elif char in ")]>a-z}":
            pair_char = char_map(char)
            pair_pos = stack.setdefault(pair_char, []).pop()

            pmap[pair_pos] = i + 1
            pmap[i] = pair_pos + 1
        elif char in "-_,:.":
            pmap[i] = 0
        else:
            raise ValueError(f"Unknown character '{char}' found in\n{dotbracket}\n")

    return pmap


def char_map(c):
    if c == ")":
        return "("
    elif c == "]":
        return "["
    elif c == ">":
        return "<"
    elif c == "}":
        return "{"
    elif c.isalpha():
        return c.upper()
    else:
        raise ValueError(f"Undefined character in charMap: {c}")


#####################
# INPUT SUBROUTINES #
#####################

def is_bpseq_file(input_file):
    with open(input_file) as f:
        for line in f:
            if not line.startswith('#'):
                if len(line.split()) != 3:
                    return False
    return True


def is_dot_bracket_file(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 2:
        sequence = lines[0].strip()
        dotbracket = lines[1].strip()
    elif len(lines) == 3:
        defline = lines[0].strip()
        sequence = lines[1].strip()
        dotbracket = lines[2].strip()
    else:
        return False

    if defline:
        if defline[0] != ">":
            return False

    if len(sequence) == len(dotbracket):
        return True
    elif " " in dotbracket:
        terms = dotbracket.split(" ")
        if len(terms) == 2:
            if len(terms[0]) == len(sequence):
                dotbracket = terms[0]
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def is_dot_bracket_file(input_file):
    with open(input_file, "r") as f:
        lines = f.readlines()

    if len(lines) == 2:
        sequence = lines[0].strip()
        dotbracket = lines[1].strip()
    elif len(lines) == 3:
        defline = lines[0].strip()
        sequence = lines[1].strip()
        dotbracket = lines[2].strip()
    else:
        return False

    if defline:
        if defline[0] != ">":
            return False

    if len(sequence) == len(dotbracket):
        return True
    elif " " in dotbracket:
        terms = dotbracket.split(" ")
        if len(terms) == 2:
            if len(terms[0]) == len(sequence):
                dotbracket = terms[0]
                return True
            else:
                return False
        else:
            return False
    else:
        return False


def read_bpseq_file(bpseq_file):
    unpaired = 0
    bp = {}
    bp_check = {}
    seq = ""

    with open(bpseq_file, 'r') as f:
        for line_number, line in enumerate(f):
            if not line.startswith('#'):
                line = line.strip()
                if len(line.split()) != 3:
                    raise ValueError(f"Bad data (need only 3 columns) on line {line_number} of {bpseq_file}:\n{line}")

                i, b, j = line.split()
                i, j = int(i), int(j)
                seq += b

                if i in bp:
                    raise ValueError(f"Fatal error: Position {i} is paired to both {bp[i]} and {j}: Line {line_number} of {bpseq_file}\n")

                if (j in bp_check) and (j != unpaired):
                    raise ValueError(f"Fatal error: Position {j} is paired to both {bp_check[j]} and {i}: Line {line_number} of {bpseq_file}\n")

                if i == j:
                    raise ValueError(f"Fatal error: Position {i} is paired to itself in {bpseq_file}: Line {line_number} of {bpseq_file}\n")

                bp[i] = j
                bp_check[j] = i

                if i in bp_check and (bp_check[i] != j):
                    raise ValueError(f"Fatal error: bpseq file at positions {i} paired to {bp_check[i]} and {j}. Caught on line {line_number} of {bpseq_file}\n")

    return bp, seq


def read_dot_bracket_file(dotbracket_file):
    with open(dotbracket_file, 'r') as f:
        lines = f.readlines()

    if len(lines) == 2:
        sequence = lines[0].strip()
        dotbracket = lines[1].strip()
    elif len(lines) == 3:
        defline = lines[0].strip()
        sequence = lines[1].strip()
        dotbracket = lines[2].strip()

    if ' ' in dotbracket:
        # could have an energy term
        terms = dotbracket.split(' ')
        dotbracket = terms[0]

    pmap = pair_map(dotbracket)
    base_pairs = {}

    for i, pair in enumerate(pmap):
        base_pairs[i + 1] = pair

    return base_pairs, sequence
    

def dot_bracket_to_structure_array(seq, dotbracket):
    # If there are basepairs
    bp = {}
    pmap = pair_map(dotbracket)
    for i, pair in enumerate(pmap):
        bp[i + 1] = pair

    if bp:
        all_segments = get_segments(bp)
        segments, knots, warnings = separate_segments(all_segments)
        bp = filter_base_pairs(bp, knots)
        segments = get_segments(bp)
        dotbracket, s, k, structure_types, page_number = build_structure_map(segments, knots, bp, seq)
        s = "".join(s)
        return s
    else:
        # Default to all external loops
        s = "E" * len(seq)
        return s
    
    
if __name__ == "__main__":
        
    # inputFile = sys.argv[1] or print(USAGE)
    inputFile = './notebooks_sRNA/str.bpseq'

    id, bp, seq = None, None, None

    if is_bpseq_file(inputFile):
        id = re.search(r"([^\/]*?)\.bpseq", inputFile).group(1)
        bp, seq = read_bpseq_file(inputFile)
    elif is_dot_bracket_file(inputFile):
        id = re.search(r"([^\/]*?)\.db", inputFile).group(1)
        bp, seq = read_dot_bracket_file(inputFile)
    else:
        raise ValueError(
            "Could not determine file type. Expecting BPSEQ or DBN (dot-bracket) file formats."
        )


    if bp:
        all_segments = get_segments(bp)
        for i, segment in enumerate(all_segments):
            if DEBUG:
                print(f"{i} {segment}")

        segments, knots, warnings = separate_segments(all_segments)
        bp = filter_base_pairs(bp, knots)
        segments = get_segments(bp)
        dotbracket, s, k, structure_types, page_number = build_structure_map(segments, knots, bp, seq)
        print_structure_types(id, seq, dotbracket, s, k, structure_types, page_number, warnings)
    else:
        dotbracket = "." * len(seq)
        s = "E" * len(seq)
        k = "N" * len(seq)
        st_file = f"{id}.st"

        with open(st_file, "w") as stf:
            stf.write(f"#Name: {id}\n")
            stf.write(f"#Length: {len(seq)}\n")
            stf.write(f"#PageNumber: 0\n")
            stf.write(f"{seq}\n{dotbracket}\n{s}\n{k}\n")
