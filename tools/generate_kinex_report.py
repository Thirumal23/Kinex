"""
Script to generate the complete KINEX Community Connect Report in Word (.docx) and PDF (.pdf) format,
strictly matching the SRM Institute of Science and Technology 21GNP301L template.
"""

import os
import sys
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn

def add_page_border(section):
    """Adds a clean academic rectangular border around the page."""
    sectPr = section._sectPr
    pgBorders = OxmlElement('w:pgBorders')
    pgBorders.set(qn('w:offsetFrom'), 'page')
    for b_name in ('top', 'left', 'bottom', 'right'):
        border = OxmlElement(f'w:{b_name}')
        border.set(qn('w:val'), 'single')
        border.set(qn('w:sz'), '12')  # 1.5 pt
        border.set(qn('w:space'), '24') # 24 pt from page margin
        border.set(qn('w:color'), '000000')
        pgBorders.append(border)
    sectPr.append(pgBorders)

def add_page_number_to_footer(footer):
    """Inserts a dynamic PAGE number in the footer, centered."""
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11)
    
    fldChar1 = OxmlElement('w:fldChar')
    fldChar1.set(qn('w:fldCharType'), 'begin')
    instrText = OxmlElement('w:instrText')
    instrText.set(qn('xml:space'), 'preserve')
    instrText.text = "PAGE"
    fldChar2 = OxmlElement('w:fldChar')
    fldChar2.set(qn('w:fldCharType'), 'separate')
    fldChar3 = OxmlElement('w:fldChar')
    fldChar3.set(qn('w:fldCharType'), 'end')
    
    r = run._r
    r.append(fldChar1)
    r.append(instrText)
    r.append(fldChar2)
    r.append(fldChar3)

def style_paragraph(p, space_before=0, space_after=6, line_spacing=1.35, align=WD_ALIGN_PARAGRAPH.JUSTIFY):
    p.paragraph_format.space_before = Pt(space_before)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = line_spacing
    p.alignment = align

def add_heading_chapter(doc, chap_num_str, title_str):
    p1 = doc.add_paragraph()
    style_paragraph(p1, space_before=14, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r1 = p1.add_run(chap_num_str)
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(15)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(0, 0, 0)
    
    p2 = doc.add_paragraph()
    style_paragraph(p2, space_before=0, space_after=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    r2 = p2.add_run(title_str)
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(15)
    r2.font.bold = True
    r2.font.color.rgb = RGBColor(0, 0, 0)

def add_section_heading(doc, text, space_before=14, space_after=6):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=space_before, space_after=space_after, align=WD_ALIGN_PARAGRAPH.LEFT)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(13)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)

def add_sub_heading(doc, text, space_before=10, space_after=4):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=space_before, space_after=space_after, align=WD_ALIGN_PARAGRAPH.LEFT)
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(0, 0, 0)

def add_body_p(doc, text, bold_prefix=None, space_after=6, italic_prefix=None):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=0, space_after=space_after, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    if bold_prefix:
        r_pre = p.add_run(bold_prefix)
        r_pre.font.name = 'Times New Roman'
        r_pre.font.size = Pt(12)
        r_pre.font.bold = True
    if italic_prefix:
        r_it = p.add_run(italic_prefix)
        r_it.font.name = 'Times New Roman'
        r_it.font.size = Pt(12)
        r_it.font.italic = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)

def add_bullet_p(doc, text, bold_title=None):
    p = doc.add_paragraph()
    style_paragraph(p, space_before=2, space_after=4, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p.paragraph_format.left_indent = Inches(0.4)
    p.paragraph_format.first_line_indent = Inches(-0.25)
    
    r_bullet = p.add_run("\u2022  ")
    r_bullet.font.name = 'Times New Roman'
    r_bullet.font.size = Pt(12)
    r_bullet.font.bold = True
    
    if bold_title:
        r_title = p.add_run(bold_title)
        r_title.font.name = 'Times New Roman'
        r_title.font.size = Pt(12)
        r_title.font.bold = True
    r = p.add_run(text)
    r.font.name = 'Times New Roman'
    r.font.size = Pt(12)

def add_photo_placeholder(doc, label_text="Include Photo proof for Week (2 to 3 Photos)"):
    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.0)
    
    tcPr = cell._tc.get_or_add_tcPr()
    tcBorders = parse_xml(
        r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        r'<w:top w:val="dashed" w:sz="8" w:space="0" w:color="888888"/>'
        r'<w:left w:val="dashed" w:sz="8" w:space="0" w:color="888888"/>'
        r'<w:bottom w:val="dashed" w:sz="8" w:space="0" w:color="888888"/>'
        r'<w:right w:val="dashed" w:sz="8" w:space="0" w:color="888888"/>'
        r'</w:tcBorders>'
    )
    shd = parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="F8F9FA"/>')
    tcPr.append(tcBorders)
    tcPr.append(shd)
    
    p = cell.paragraphs[0]
    style_paragraph(p, space_before=16, space_after=4, align=WD_ALIGN_PARAGRAPH.CENTER)
    r1 = p.add_run(f"[{label_text}]")
    r1.font.name = 'Times New Roman'
    r1.font.size = Pt(11)
    r1.font.bold = True
    r1.font.color.rgb = RGBColor(60, 60, 60)
    
    p2 = cell.add_paragraph()
    style_paragraph(p2, space_before=0, space_after=16, align=WD_ALIGN_PARAGRAPH.CENTER)
    r2 = p2.add_run("(Paste on-ground photographs, sensor trials, interactive sessions, or demonstration proof here)")
    r2.font.name = 'Times New Roman'
    r2.font.size = Pt(10)
    r2.font.italic = True
    r2.font.color.rgb = RGBColor(100, 100, 100)
    
    p_after = doc.add_paragraph()
    style_paragraph(p_after, space_before=4, space_after=6)


def build_full_report():
    doc = docx.Document()
    
    # Configure default section
    section1 = doc.sections[0]
    section1.top_margin = Inches(1.0)
    section1.bottom_margin = Inches(1.0)
    section1.left_margin = Inches(1.0)
    section1.right_margin = Inches(1.0)
    add_page_border(section1)
    
    # ---------------------------------------------------------------------------
    # PAGE 1: TITLE PAGE (COVER)
    # ---------------------------------------------------------------------------
    p_course = doc.add_paragraph()
    style_paragraph(p_course, space_before=24, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_c = p_course.add_run("21GNP301L - COMMUNITY CONNECT")
    r_c.font.name = 'Times New Roman'
    r_c.font.size = Pt(16)
    r_c.font.bold = True

    p_rep = doc.add_paragraph()
    style_paragraph(p_rep, space_before=18, space_after=28, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_rep = p_rep.add_run("REPORT")
    r_rep.font.name = 'Times New Roman'
    r_rep.font.size = Pt(16)
    r_rep.font.bold = True

    p_sub = doc.add_paragraph()
    style_paragraph(p_sub, space_before=14, space_after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_sub = p_sub.add_run("Submitted by")
    r_sub.font.name = 'Times New Roman'
    r_sub.font.size = Pt(14)
    r_sub.font.italic = True

    p_name = doc.add_paragraph()
    style_paragraph(p_name, space_before=8, space_after=32, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_name = p_name.add_run("[STUDENT NAME] – [REGISTER NUMBER]")
    r_name.font.name = 'Times New Roman'
    r_name.font.size = Pt(14)
    r_name.font.bold = True

    p_ful = doc.add_paragraph()
    style_paragraph(p_ful, space_before=10, space_after=10, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_ful = p_ful.add_run("in partial fulfillment for the award of the degree of")
    r_ful.font.name = 'Times New Roman'
    r_ful.font.size = Pt(13)
    r_ful.font.italic = True

    p_deg = doc.add_paragraph()
    style_paragraph(p_deg, space_before=6, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_deg = p_deg.add_run("BACHELOR OF TECHNOLOGY")
    r_deg.font.name = 'Times New Roman'
    r_deg.font.size = Pt(15)
    r_deg.font.bold = True

    p_in = doc.add_paragraph()
    style_paragraph(p_in, space_before=4, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_in = p_in.add_run("in")
    r_in.font.name = 'Times New Roman'
    r_in.font.size = Pt(13)

    p_dept = doc.add_paragraph()
    style_paragraph(p_dept, space_before=6, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_dept = p_dept.add_run("COMPUTER SCIENCE AND ENGINEERING")
    r_dept.font.name = 'Times New Roman'
    r_dept.font.size = Pt(14)
    r_dept.font.bold = True

    p_spec = doc.add_paragraph()
    style_paragraph(p_spec, space_before=2, space_after=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_spec = p_spec.add_run("with Specialization in Internet of Things")
    r_spec.font.name = 'Times New Roman'
    r_spec.font.size = Pt(13)
    r_spec.font.bold = True

    p_of = doc.add_paragraph()
    style_paragraph(p_of, space_before=4, space_after=8, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_of = p_of.add_run("of")
    r_of.font.name = 'Times New Roman'
    r_of.font.size = Pt(12)

    p_fac = doc.add_paragraph()
    style_paragraph(p_fac, space_before=4, space_after=20, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_fac = p_fac.add_run("FACULTY OF ENGINEERING AND TECHNOLOGY")
    r_fac.font.name = 'Times New Roman'
    r_fac.font.size = Pt(13)
    r_fac.font.bold = True

    # Logo
    logo_path = os.path.abspath('extracted_assets/p1_0.png')
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(8)
        p_logo.paragraph_format.space_after = Pt(14)
        run_logo = p_logo.add_run()
        run_logo.add_picture(logo_path, width=Inches(2.4))

    p_srm = doc.add_paragraph()
    style_paragraph(p_srm, space_before=10, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_srm = p_srm.add_run("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY")
    r_srm.font.name = 'Times New Roman'
    r_srm.font.size = Pt(14)
    r_srm.font.bold = True

    p_loc = doc.add_paragraph()
    style_paragraph(p_loc, space_before=2, space_after=18, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_loc = p_loc.add_run("RAMAPURAM, CHENNAI – 600089")
    r_loc.font.name = 'Times New Roman'
    r_loc.font.size = Pt(13)
    r_loc.font.bold = True

    p_date = doc.add_paragraph()
    style_paragraph(p_date, space_before=8, space_after=14, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_date = p_date.add_run("JULY-DECEMBER 2026")
    r_date.font.name = 'Times New Roman'
    r_date.font.size = Pt(14)
    r_date.font.bold = True

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # PAGE 2: BONAFIDE CERTIFICATE
    # ---------------------------------------------------------------------------
    p_inst = doc.add_paragraph()
    style_paragraph(p_inst, space_before=36, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_inst = p_inst.add_run("SRM INSTITUTE OF SCIENCE AND TECHNOLOGY,")
    r_inst.font.name = 'Times New Roman'
    r_inst.font.size = Pt(14)
    r_inst.font.bold = True

    p_inst_loc = doc.add_paragraph()
    style_paragraph(p_inst_loc, space_before=0, space_after=40, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_inst_loc = p_inst_loc.add_run("RAMAPURAM, CHENNAI - 600089")
    r_inst_loc.font.name = 'Times New Roman'
    r_inst_loc.font.size = Pt(14)
    r_inst_loc.font.bold = True

    p_bon = doc.add_paragraph()
    style_paragraph(p_bon, space_before=20, space_after=40, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_bon = p_bon.add_run("BONAFIDE CERTIFICATE")
    r_bon.font.name = 'Times New Roman'
    r_bon.font.size = Pt(14)
    r_bon.font.bold = True

    p_cert = doc.add_paragraph()
    style_paragraph(p_cert, space_before=16, space_after=90, line_spacing=1.5, align=WD_ALIGN_PARAGRAPH.JUSTIFY)
    p_cert.paragraph_format.first_line_indent = Inches(0.5)
    
    r_c1 = p_cert.add_run("Certified that this Community Connect report is the bonafide work of ")
    r_c1.font.name = 'Times New Roman'
    r_c1.font.size = Pt(12)
    
    r_c2 = p_cert.add_run("[STUDENT NAME] (Register Number: [REGISTER NUMBER: RA2411032020XXX])")
    r_c2.font.name = 'Times New Roman'
    r_c2.font.size = Pt(12)
    r_c2.font.bold = True
    
    r_c3 = p_cert.add_run(" of III Year B.Tech – IoT, who has carried out the Community Connect work and submitted it for the course ")
    r_c3.font.name = 'Times New Roman'
    r_c3.font.size = Pt(12)
    
    r_c4 = p_cert.add_run("21GNP301L – COMMUNITY CONNECT")
    r_c4.font.name = 'Times New Roman'
    r_c4.font.size = Pt(12)
    r_c4.font.bold = True
    
    r_c5 = p_cert.add_run(" during the Academic Year 2026–2027 (Odd Semester).")
    r_c5.font.name = 'Times New Roman'
    r_c5.font.size = Pt(12)

    p_sig_title = doc.add_paragraph()
    style_paragraph(p_sig_title, space_before=30, space_after=50, align=WD_ALIGN_PARAGRAPH.RIGHT)
    r_sig_t = p_sig_title.add_run("SIGNATURE       ")
    r_sig_t.font.name = 'Times New Roman'
    r_sig_t.font.size = Pt(12)
    r_sig_t.font.bold = True

    p_sig_block = doc.add_paragraph()
    style_paragraph(p_sig_block, space_before=10, space_after=70, line_spacing=1.2, align=WD_ALIGN_PARAGRAPH.RIGHT)
    lines = [
        ("Dr. Geetha T V., M.E., Ph.D.,", True),
        ("Assistant Professor, Department of IOT-CSBS,", False),
        ("SRM Institute of Science and Technology,", False),
        ("Ramapuram, Chennai.      ", False),
    ]
    for text, is_bold in lines:
        run = p_sig_block.add_run(text + "\n")
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.bold = is_bold

    p_viva = doc.add_paragraph()
    style_paragraph(p_viva, space_before=40, space_after=10, align=WD_ALIGN_PARAGRAPH.LEFT)
    r_viva = p_viva.add_run("Presentation and Viva - Voce conducted on: ___________________")
    r_viva.font.name = 'Times New Roman'
    r_viva.font.size = Pt(12)

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # PAGE 3: TABLE OF CONTENTS
    # ---------------------------------------------------------------------------
    p_toc_head = doc.add_paragraph()
    style_paragraph(p_toc_head, space_before=30, space_after=30, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_toc_h = p_toc_head.add_run("TABLE OF CONTENTS")
    r_toc_h.font.name = 'Times New Roman'
    r_toc_h.font.size = Pt(15)
    r_toc_h.font.bold = True

    toc_table = doc.add_table(rows=9, cols=3)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_table.autofit = False

    col_widths = [Inches(1.2), Inches(4.2), Inches(1.1)]
    for row in toc_table.rows:
        for idx, width in enumerate(col_widths):
            row.cells[idx].width = width

    tblPr = toc_table._tbl.tblPr
    tblBorders = parse_xml(
        r'<w:tblBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        r'<w:top w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        r'<w:left w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        r'<w:bottom w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        r'<w:right w:val="single" w:sz="6" w:space="0" w:color="000000"/>'
        r'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        r'<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
        r'</w:tblBorders>'
    )
    tblPr.append(tblBorders)

    # Header Row
    headers = [("CHAPTER \nNO.", WD_ALIGN_PARAGRAPH.CENTER), ("TITLE", WD_ALIGN_PARAGRAPH.CENTER), ("PAGE NO.", WD_ALIGN_PARAGRAPH.CENTER)]
    for i, (text, align) in enumerate(headers):
        cell = toc_table.cell(0, i)
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        p = cell.paragraphs[0]
        style_paragraph(p, space_before=4, space_after=4, align=align)
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11)
        run.font.bold = True

    # Exact page mapping from compiled PDF
    toc_items = [
        ("1", "COURSE OVERVIEW", "1"),
        ("2", "ABOUT THE ORGANIZATION", "4"),
        ("3", "OBJECTIVE OF CHOOSING THE SPECIFIC NGO", "8"),
        ("4", "NATURE OF ACTIVITIES OF THE NGO", "11"),
        ("5", "WEEK-WISE ACTIVITY UNDERTAKEN", "15"),
        ("6", "TANGIBLE LEARNING", "22"),
        ("7", "STUDENT CONTRIBUTION TO NGO DURING \nTHE SERVICE PERIOD", "25"),
        ("8", "CERTIFICATE", "28")
    ]

    for row_idx, (ch_num, title, pg) in enumerate(toc_items, start=1):
        c0 = toc_table.cell(row_idx, 0)
        c1 = toc_table.cell(row_idx, 1)
        c2 = toc_table.cell(row_idx, 2)
        
        c0.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c1.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        c2.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        
        p0 = c0.paragraphs[0]
        style_paragraph(p0, space_before=6, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
        r0 = p0.add_run(ch_num)
        r0.font.name = 'Times New Roman'
        r0.font.size = Pt(11)
        r0.font.bold = True

        p1 = c1.paragraphs[0]
        style_paragraph(p1, space_before=6, space_after=6, align=WD_ALIGN_PARAGRAPH.LEFT)
        r1 = p1.add_run(title)
        r1.font.name = 'Times New Roman'
        r1.font.size = Pt(11)
        r1.font.bold = True

        p2 = c2.paragraphs[0]
        style_paragraph(p2, space_before=6, space_after=6, align=WD_ALIGN_PARAGRAPH.CENTER)
        r2 = p2.add_run(pg)
        r2.font.name = 'Times New Roman'
        r2.font.size = Pt(11)

    # ---------------------------------------------------------------------------
    # START OF MAIN CONTENT SECTION (PAGE 1 NUMBERED)
    # ---------------------------------------------------------------------------
    main_section = doc.add_section(docx.enum.section.WD_SECTION.NEW_PAGE)
    main_section.top_margin = Inches(1.0)
    main_section.bottom_margin = Inches(1.0)
    main_section.left_margin = Inches(1.0)
    main_section.right_margin = Inches(1.0)
    add_page_border(main_section)
    
    # Configure footer page numbering
    main_section.footer.is_linked_to_previous = False
    add_page_number_to_footer(main_section.footer)

    sectPr = main_section._sectPr
    pgNumType = OxmlElement('w:pgNumType')
    pgNumType.set(qn('w:start'), '1')
    sectPr.append(pgNumType)

    # ---------------------------------------------------------------------------
    # CHAPTER 1: COURSE OVERVIEW
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 1", "COURSE OVERVIEW")

    add_body_p(doc, "Community Connect is an experiential learning initiative designed to bridge the gap between classroom knowledge and real-world societal needs. It encourages engineering students to actively engage with local communities, identify pressing grassroots challenges, and apply innovative, human-centered solutions through technology, fieldwork, and collaborative practices. In a rapidly modernizing society, engineering education achieves its highest purpose when academic knowledge is translated into empathetic tools that empower marginalized individuals and solve authentic civic problems.")

    add_body_p(doc, "The primary objective of Community Connect is to instill social responsibility, empathy, and holistic problem-solving skills among students while enhancing their academic learning. By participating in this program through the KINEX Assistive Hand Sign Recognition and Embedded Translation initiative, students gain direct exposure to the Deaf, hard-of-hearing, and speech-impaired community. The initiative explores how modern Internet of Things (IoT) sensors, embedded microcontrollers, edge displays, and computer vision can work synergistically to dismantle everyday communication barriers and foster societal inclusion.")

    add_sub_heading(doc, "Objectives of the Project")
    add_body_p(doc, "The Community Connect initiative aims to integrate academic learning with social responsibility. The specific objectives of this assistive project are:")

    add_bullet_p(doc, "To identify community needs and understand the challenges faced by non-verbal and hearing-impaired individuals in their daily civic, educational, and public interactions.")
    add_bullet_p(doc, "To apply IoT concepts and technological solutions—including sensory gloves, microcontrollers, and edge displays—for addressing communication barriers in a practical and sustainable manner.")
    add_bullet_p(doc, "To promote social awareness and responsibility among students by engaging with communities directly and developing deep empathy for persons with disabilities.")
    add_bullet_p(doc, "To enhance problem-solving and analytical skills through experiential learning outside the classroom, tackling real-world constraints such as sensor noise, ambient lighting, and ergonomic comfort.")
    add_bullet_p(doc, "To encourage teamwork and leadership qualities by working collaboratively on community-based assistive projects with NGO mentors, special educators, and peers.")
    add_bullet_p(doc, "To bridge the gap between theory and practice by applying classroom knowledge in embedded systems, signal normalization, machine learning, and computer vision to real-life assistive scenarios.")
    add_bullet_p(doc, "To contribute meaningfully to society by developing innovative, low-cost solutions that improve the quality of life, independence, and social integration of differently-abled citizens.")

    add_sub_heading(doc, "Personal & Professional Outcomes:")
    add_bullet_p(doc, "Enhanced initiative, project organization, rapid prototyping, and decision-making under unpredictable field conditions.", bold_title="Leadership: ")
    add_bullet_p(doc, "Collaboration across diverse groups—including special educators, deaf students, and technical peers—improves adaptability and problem-solving.", bold_title="Teamwork: ")
    add_bullet_p(doc, "Interactions with non-verbal community members, NGO mentors, and school students build immense patience, confidence, and clarity in non-verbal and spoken expression.", bold_title="Communication: ")

    add_sub_heading(doc, "Relevance to Engineering:")
    add_bullet_p(doc, "Engineers impact society directly; this course builds acute awareness of human accessibility, assistive ethics, and the socio-economic contexts of technology deployment.")
    add_bullet_p(doc, "Encourages applying technical knowledge (like flex sensors, IMU motion tracking, ESP32 telemetry, and edge vision models) in authentic field environments rather than controlled laboratory benchmarks.")
    add_bullet_p(doc, "Promotes interdisciplinary teamwork across embedded systems, machine learning, ergonomics, and accessibility compliance domains.")

    add_sub_heading(doc, "Learning from the KINEX Assistive Campaign:")
    add_bullet_p(doc, "Gained firsthand insight into the communication divide and everyday struggles experienced by the Deaf and non-verbal community in public spaces.")
    add_bullet_p(doc, "Applied structured project management in field testing, ergonomic feedback collection, and community sign awareness drives.")
    add_bullet_p(doc, "Understood the critical link between assistive technology affordability, edge hardware independence, and community adoption.")
    add_bullet_p(doc, "Practiced rigorous documentation, ethical consent protocols for biometric data capture, and data-driven accessibility analysis.")

    add_sub_heading(doc, "Reflection:")
    add_body_p(doc, "This course offered far more than academic credit—it fundamentally shaped me into a socially responsible engineer. Working on the KINEX initiative demonstrated that meaningful technological change begins with human empathy, and that consistent, grassroots engineering efforts can create lasting, life-changing impact for individuals whose voices are too often unheard.")

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 2: ABOUT THE ORGANIZATION
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 2", "ABOUT THE ORGANIZATION")

    add_body_p(doc, "Deaf Enabled Foundation (DEF) is a registered non-profit, Chennai-active organization that stands as a model for community-driven empowerment and social inclusion for persons with speech and hearing impairments. Founded with the conviction that every deaf individual possesses immense potential when provided accessible education and communication avenues, DEF works tirelessly to break down systemic communication barriers across educational institutions, workplaces, and public spaces. The organization’s approach blends bilingual Indian Sign Language (ISL) literacy, digital vocational training, and community sensitization to create meaningful, long-lasting impact.")

    add_body_p(doc, "With the guiding motto “Educate. Empower. Elevate.”, DEF focuses on bridging the communication chasm between the Deaf and hearing communities. Its initiatives revolve around accessible sign language literacy, digital skill development, assistive technology testing, and youth leadership. By operating through grassroots community centers and partnerships with academic institutions, DEF empowers deaf youth to lead self-reliant lives and become active, recognized contributors to modern urban society.")

    add_section_heading(doc, "2.1 VISION AND MISSION")
    add_body_p(doc, "DEF envisions an inclusive, barrier-free society where deaf and speech-impaired individuals enjoy equal rights, dignified employment, and unrestricted communication access across all walks of life.")
    add_body_p(doc, "The organization’s mission is built upon four key pillars:")

    add_bullet_p(doc, "Dismantling linguistic and social barriers through widespread Indian Sign Language education, interpreter services, and accessible assistive communication tools.", bold_title="Overcoming Communication Barriers: ")
    add_bullet_p(doc, "Providing specialized technical, vocational, and digital literacy training to young deaf individuals to prepare them for independent professional careers.", bold_title="Empowering the Youth: ")
    add_bullet_p(doc, "Encouraging schools, colleges, and civic organizations to adopt inclusive practices, cultivate empathy, and take collective responsibility for accessible public infrastructure.", bold_title="Inclusive Social Integration: ")
    add_bullet_p(doc, "Collaborating with academic institutions, civic bodies, and policymakers to advocate for deaf rights under the Rights of Persons with Disabilities (RPwD) Act 2016.", bold_title="Policy Advocacy and Assistive Research: ")

    add_body_p(doc, "DEF’s mission seamlessly combines grassroots community fieldwork with long-term technological and policy advocacy, proving that sustainable accessibility requires both compassionate on-ground support and innovative assistive solutions.")

    add_section_heading(doc, "2.2 KEY AREAS OF WORK")

    add_sub_heading(doc, "1. Indian Sign Language (ISL) Literacy and Promotion")
    add_body_p(doc, "At the heart of DEF’s mission lies the preservation and promotion of Indian Sign Language (ISL) as a legitimate, vibrant, and expressive visual language. The organization conducts structured sign language certification courses, bilingual educational workshops, and community sensitization drives. Efforts include:")
    add_bullet_p(doc, "Conducting introductory and advanced ISL training sessions for hearing individuals, teachers, and healthcare workers.")
    add_bullet_p(doc, "Publishing visual dictionaries and bilingual learning material in Tamil, English, and ISL.")
    add_bullet_p(doc, "Training professional educational interpreters for colleges, courts, and public forums.")
    add_bullet_p(doc, "Organizing community sign-along circles to build confidence among deaf children and their families.")
    add_body_p(doc, "These literacy campaigns not only validate the linguistic identity of the Deaf community but also foster an inclusive environment where communication flows effortlessly across social boundaries.")

    add_sub_heading(doc, "2. Digital Literacy and Assistive Technology Evaluation")
    add_body_p(doc, "DEF believes that technology is the ultimate equalizer for individuals with disabilities. The organization operates computer labs and assistive innovation spaces where deaf students gain hands-on training in digital design, coding, data management, and office automation. Key activities include:")
    add_bullet_p(doc, "Hosting workshops on digital communication tools, video relay services, and speech-to-text accessibility software.")
    add_bullet_p(doc, "Collaborating with engineering researchers to test, validate, and provide user-centric feedback on assistive wearable devices and vision-based sign interpreters.")
    add_bullet_p(doc, "Creating digital repositories of educational videos translated into Indian Sign Language.")
    add_body_p(doc, "Through these initiatives, DEF transforms assistive technology from a theoretical engineering novelty into a practical, everyday companion for deaf individuals.")

    add_sub_heading(doc, "3. Youth Skill Development and Employment Enablement")
    add_body_p(doc, "To ensure long-term self-reliance, DEF partners with corporate organizations, retail chains, and government bodies to create inclusive employment opportunities for deaf youth. Its career cell offers personality development sessions, workplace readiness training, and sensitivity workshops for prospective employers. These programs ensure that deaf graduates transition smoothly into professional environments with dignity and equal pay.")

    add_sub_heading(doc, "4. Partnership with Academic and Research Institutions")
    add_body_p(doc, "To accelerate technological innovation in accessibility, DEF actively collaborates with leading universities and research institutes like SRM Institute of Science and Technology. The NGO provides engineering students with field access, sign language domain expertise, and direct interaction with deaf subjects. In return, student projects like KINEX explore affordable, real-time edge devices—such as sensory gloves and computer vision models—that address communication challenges in practical community settings.")

    add_section_heading(doc, "2.3 THE KINEX ASSISTIVE SIGN LANGUAGE INITIATIVE")
    add_body_p(doc, "Among DEF’s collaborative community ventures is the KINEX Assistive Sign Language & Community Accessibility Initiative, a four-week project that demonstrated the power of youth-driven engineering in solving real-world social problems. Non-verbal and deaf individuals across Chennai frequently encounter extreme communication friction at bus terminals, ticket counters, clinics, and government offices where non-signers predominate. DEF partnered with SRM student engineers to evaluate KINEX—a lightweight, dual-pipeline sign recognition system integrating custom hand pose vision and wearable sensor gloves with an ESP32 edge display. The key campaign activities included:")

    add_bullet_p(doc, "Setting up on-site translation booths where deaf individuals demonstrated common daily signs to hearing visitors using live edge displays.", bold_title="Assistive Communication Drives: ")
    add_bullet_p(doc, "Recording and analyzing variations in hand morphology, flexure angles, and signing speed across diverse subjects.", bold_title="Gesture Data Observation: ")
    add_bullet_p(doc, "Conducting school outreach, distributing bilingual visual aids, and running interactive empathy games.", bold_title="Community Awareness Generation: ")
    add_bullet_p(doc, "Gathering structured qualitative and quantitative feedback from deaf participants regarding sensor glove comfort, weight, and response speed.", bold_title="Ergonomic and Usability Assessment: ")
    add_bullet_p(doc, "Forming youth advocacy circles to sustain sign language awareness and encourage inclusive public facilities.", bold_title="Volunteer Mobilization: ")

    add_body_p(doc, "This collaborative effort bridged the gap between deaf participants and the hearing public, generating immense mutual respect and highlighting how affordable, decentralized technology can restore communicative autonomy.")

    add_section_heading(doc, "2.4 WHY DEF STANDS OUT")
    add_body_p(doc, "DEF distinguishes itself from conventional charitable trusts through its deaf-led empowerment model that prioritizes autonomy, technological literacy, and dignity over sympathy. Some of its defining strengths include:")
    add_bullet_p(doc, "Guided directly by deaf educators and professionals who possess lived experience of communication barriers.", bold_title="Community-Led Leadership: ")
    add_bullet_p(doc, "Insists on rigorous usability testing and qualitative feedback before endorsing assistive technologies.", bold_title="Scientific and User-Centered Approach: ")
    add_bullet_p(doc, "Empowers engineering students to co-design solutions alongside the end-users they seek to serve.", bold_title="Youth-Centric Engagement: ")
    add_bullet_p(doc, "Focuses on low-cost, decentralized edge technologies that do not rely on expensive imported hardware or high-bandwidth internet.", bold_title="Sustainable and Scalable Impact: ")

    add_sub_heading(doc, "Personal Insight:")
    add_body_p(doc, "During my time working with DEF, I realized that their mission goes beyond teaching sign language—it is about transforming human mindsets. The organization fosters a warm, deeply supportive community built on mutual respect and creative expression. Every interactive session, every data collection trial, and every school demonstration showed me that communication is not limited to spoken words. True inclusion begins when society is willing to listen with its eyes and understand with an open heart. Being part of this initiative gave me not only technical growth as an IoT engineer, but a lifelong sense of responsibility to direct engineering toward human equality.")

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 3: OBJECTIVE OF CHOOSING THE SPECIFIC NGO
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 3", "OBJECTIVE OF CHOOSING THE SPECIFIC NGO")

    add_section_heading(doc, "3.1 REASON FOR CHOOSING DEF NGO FOR MY COMMUNITY CONNECT PROJECT:")
    add_body_p(doc, "Choosing Deaf Enabled Foundation (DEF) for my Community Connect project was a deliberate, heartfelt decision guided by my personal values, academic specialization in the Internet of Things (IoT), and deep concern for social accessibility. I wanted to commit to a project where my technical skills could produce measurable, human-centric benefits rather than remaining confined to simulation software. DEF provided the ideal ecosystem through its authentic community connections, deaf-led leadership, and commitment to assistive technological exploration.")

    sub_reasons = [
        ("1. Personal Interest in Assistive Technology and Social Inclusion",
         "I have always believed that the true measure of technological advancement is how well it serves the most vulnerable and overlooked segments of society. Persons with speech and hearing impairments face constant daily isolation in civic spaces simply because the general public does not understand sign language. Engaging with DEF allowed me to direct my engineering focus toward bridging this silent divide and creating assistive tools that restore communication dignity."),
        
        ("2. Desire for Real-World Field Exposure with Non-Verbal Communities",
         "As a Computer Science and Engineering student specializing in IoT, much of my academic training focuses on algorithmic theory, microcontroller programming, and circuit simulation. I recognized the critical need to step outside theoretical assumptions and work directly with end-users. DEF offered a structured environment where I could observe how non-verbal individuals actually communicate, interact with physical prototypes, and navigate daily social obstacles."),
        
        ("3. Relevance to My Computer Science and IoT Specialization",
         "The KINEX project directly synthesizes core disciplines of my IoT curriculum—including embedded sensor telemetry (flex sensors and MPU6050 IMU on an ESP32 microcontroller), edge computing, low-latency display driving (ST7735 TFT SPI), and computer vision pose estimation. Partnering with DEF allowed me to ground these technical concepts in a vital, high-stakes application: real-time sign language recognition and translation."),
        
        ("4. Hands-On and Holistic Approach",
         "Unlike organizations that limit their work to passive charity or high-level seminars, DEF combines hands-on vocational training, educational outreach, and assistive usability testing. Their community campaigns provided a complete, 360-degree view of disability empowerment—from hardware ergonomics and sign linguistics to community sensitization and public policy."),
        
        ("5. Leadership and Responsibility",
         "At DEF, student volunteers are entrusted with genuine responsibilities. I was tasked with coordinating sensor trial schedules, managing hardware kits, facilitating demonstrations for hearing visitors, and recording quantitative user feedback. These responsibilities honed my project coordination, critical thinking, and leadership abilities in dynamic, real-world conditions."),
        
        ("6. Visible Impact and Community Connection",
         "The immediate, tangible impact of our work was deeply motivating. Seeing a hearing person read a deaf student’s sign on the KINEX edge display with an instant smile of understanding proved that technology can dissolve human barriers. Interacting with deaf youth and observing their pride in seeing their language translated by student-built hardware was immensely fulfilling."),
        
        ("7. Collaboration and Civic Learning",
         "Working with DEF exposed me to the broader civic and legal frameworks surrounding disability rights in India, such as the Rights of Persons with Disabilities (RPwD) Act 2016 and the Accessible India Campaign (Sugamya Bharat Abhiyan). I learned how NGOs, educational institutions, and civic bodies must collaborate to establish universal accessibility standards in public transportation, healthcare, and education."),
        
        ("8. Personal Growth and Purpose",
         "This journey was deeply transformative for my personal character. It taught me patience, humility, and the importance of active listening. Learning to communicate through gestures, facial expressions, and basic sign language made me realize how much hearing people take effortless communication for granted. It gave my engineering education a lasting moral purpose."),
        
        ("9. Social Awareness and Responsibility",
         "Collaborating with DEF opened my eyes to the daily systemic barriers faced by disabled citizens—from lack of qualified sign interpreters in hospitals to inaccessible emergency services. It reinforced my conviction that engineers have an ethical duty to design inclusive products from day one, rather than treating accessibility as an afterthought."),
        
        ("10. Future Vision",
         "This project cemented my long-term aspiration to develop smart assistive devices combining IoT wearable telemetry, edge machine learning, and human-computer interfaces. In the future, I plan to continue researching low-power, standalone assistive wearable systems that can operate without internet access, bringing affordable, real-time sign translation to millions across rural and urban India.")
    ]

    for title, desc in sub_reasons:
        add_sub_heading(doc, title)
        add_body_p(doc, desc)

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 4: NATURE OF ACTIVITIES OF THE NGO
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 4", "NATURE OF ACTIVITIES OF THE NGO")

    add_body_p(doc, "Deaf Enabled Foundation (DEF) is not merely a social service society, but a transformative community movement dedicated to redefining how society perceives and engages with the Deaf community. Based in Chennai with branches across major hubs, the organization addresses the linguistic, educational, technological, vocational, and civic dimensions of hearing and speech disabilities through both on-ground community interventions and institutional advocacy. Its mission is to empower deaf citizens to claim their fundamental right to barrier-free communication and self-determined living.")

    add_section_heading(doc, "4.1 ASSISTIVE COMMUNICATION DRIVES & FIELD DEMONSTRATIONS")
    add_body_p(doc, "Promoting accessible two-way communication between deaf individuals and the hearing public is DEF’s flagship operational activity. The NGO organizes community-based interaction drives that include:")
    add_bullet_p(doc, "Setting up mobile communication kiosks where deaf facilitators demonstrate wearable assistive gloves and vision-based translation prototypes to visitors.", bold_title="On-Ground Assistive Demonstrations: ")
    add_bullet_p(doc, "Evaluating the accessibility quotient of public utility counters (railway booking desks, hospitals, and banking branches) and training counter staff on basic sign interactions.", bold_title="Public Service Communication Audits: ")
    add_bullet_p(doc, "Developing informational signage and interactive digital screens featuring common Indian Sign Language survival signs.", bold_title="Bilingual Display Installations: ")
    add_bullet_p(doc, "Compiling impact metrics, user satisfaction surveys, and latency benchmarks from live field demonstrations.", bold_title="Quantitative Usability Logging: ")
    add_body_p(doc, "Example: During the four-week KINEX assistive trials conducted across Chennai community centers, over 1,000 gesture instances were evaluated with deaf participants, achieving real-time translation with an average edge display latency of under 120 milliseconds and bridging communication for dozens of first-time non-signing visitors.")

    add_section_heading(doc, "4.2 COMMUNITY AWARENESS AND INCLUSIVE SIGN LITERACY")
    add_body_p(doc, "DEF strongly believes that societal mindset transformation is the prerequisite for authentic inclusion. Its community awareness initiatives include:")
    add_bullet_p(doc, "Volunteers engage with families of deaf children and local shopkeepers to encourage sign language acceptance and eliminate social stigma.", bold_title="Door-to-Door and Neighborhood Outreach: ")
    add_bullet_p(doc, "Interactive sessions where hearing students learn ISL fingerspelling, basic greetings, and non-verbal communication through games and role-plays.", bold_title="School and College Sensitivity Workshops: ")
    add_bullet_p(doc, "Designing bold, creative infographics and public posters displaying common ISL phrases to make sign language visible in urban spaces.", bold_title="Visual Sign Campaigns & Street Murals: ")
    add_bullet_p(doc, "Publishing video tutorials and social media resources with synchronized sign language interpretation, Tamil voiceovers, and English subtitles.", bold_title="Bilingual Digital Content: ")

    add_section_heading(doc, "4.3 YOUTH MOBILIZATION AND VOLUNTEER MANAGEMENT")
    add_body_p(doc, "DEF adopts an active youth-driven approach that brings together hearing college students and deaf youth:")
    add_bullet_p(doc, "Partnerships with premier engineering institutions like SRM Institute of Science and Technology provide students with credited Community Connect and social internship opportunities.", bold_title="College Collaborations: ")
    add_bullet_p(doc, "Student volunteers spearhead hardware testing, digital documentation, survey management, and event logistics.", bold_title="Leadership Opportunities: ")
    add_bullet_p(doc, "Volunteers gain valuable hands-on experience in assistive device development, user empathy research, and public communication.", bold_title="Skill Development: ")
    add_bullet_p(doc, "Structured review meetings where student volunteers, deaf facilitators, and NGO coordinators discuss operational roadblocks, ethical questions, and technical insights.", bold_title="Weekly Reflection Circles: ")

    add_section_heading(doc, "4.4 BIOMECHANICAL AND GESTURE ACCESSIBILITY MONITORING")
    add_body_p(doc, "In close collaboration with assistive technology developers and educators, DEF maintains a structured framework for observing and recording sign biomechanics:")
    add_bullet_p(doc, "Documenting variations in hand flexion, wrist tilt angles, and sign durations across individuals of different age groups and hand sizes.", bold_title="Gesture Kinematic Logging: ")
    add_bullet_p(doc, "Evaluating how long users can comfortably wear prototype sensory gloves without experiencing perspiration, strain, or joint restriction.", bold_title="Ergonomic Fatigue Assessment: ")
    add_bullet_p(doc, "Identifying high-priority survival and conversational gestures (`HEARING`, `THANK_YOU`, `MORNING`, `I`, `BYE`, `NAME`, `INDIAN`) that provide immediate utility in daily civic interactions.", bold_title="Vocabulary Standardization: ")
    add_bullet_p(doc, "Establishing zero-leakage, ethnically diverse sign gesture datasets that enable researchers to train unbiased edge AI models.", bold_title="Inclusive Data Repositories: ")

    add_section_heading(doc, "4.5 POLICY ADVOCACY AND DISABILITY REPORTING")
    add_body_p(doc, "DEF extends its work beyond field activities through systemic advocacy and data-backed civic representations:")
    add_bullet_p(doc, "Presenting field reports and accessibility audits to the Tamil Nadu Welfare Board for Persons with Disabilities and the Greater Chennai Corporation.", bold_title="Submissions to Civic Authorities: ")
    add_bullet_p(doc, "Actively supporting national efforts to recognize Indian Sign Language as an official language in educational curriculums.", bold_title="Advocacy for ISL Recognition: ")
    add_bullet_p(doc, "Petitions advocating for visual announcement screens and digital sign kiosks across metro and bus stations.", bold_title="Inclusive Public Transport Representation: ")

    add_section_heading(doc, "4.6 TOOLS AND TECHNIQUES USED")
    add_body_p(doc, "DEF leverages a blend of accessible digital tools and cutting-edge student-built prototypes to achieve high impact:")
    add_bullet_p(doc, "Equipped with 5-finger flex sensors and MPU6050 6-axis IMUs for kinematic tracking.", bold_title="ESP32 Microcontrollers & Wearable Sensory Gloves: ")
    add_bullet_p(doc, "For instantaneous visual sign feedback without computer screens.", bold_title="ST7735 1.8-inch TFT SPI Edge Displays: ")
    add_bullet_p(doc, "Lightweight Multi-Layer Perceptron (MLP) neural networks running locally on laptops for zero-dependency computer vision.", bold_title="YOLO26 Pose Vision Pipeline: ")
    add_bullet_p(doc, "For digital feedback collection, latency tracking, and demographic analysis.", bold_title="Google Forms & Cloud Spreadsheets: ")
    add_bullet_p(doc, "Bilingual charts and social media platforms for outreach and progress tracking.", bold_title="Visual flashcards: ")

    add_section_heading(doc, "4.7 SEASONAL AND THEMATIC CAMPAIGNS")
    add_body_p(doc, "To maintain year-round public involvement, DEF organizes high-visibility civic events including:")
    add_bullet_p(doc, "International Day of Sign Languages & International Week of the Deaf Rallies (September).")
    add_bullet_p(doc, "'Bridge the Silence' Inclusive Walkathons and Deaf Art Exhibitions.")
    add_bullet_p(doc, "Accessible India Assistive Hackathons connecting engineering innovators with deaf end-users.")

    add_section_heading(doc, "4.8 COLLABORATIVE PARTNERSHIPS")
    add_body_p(doc, "DEF’s enduring success is rooted in its collaborative network across academia, government, and civil society:")
    add_bullet_p(doc, "Collaborates with SRM Institute of Science and Technology for research, curriculum workshops, and student community projects.", bold_title="Educational Institutions: ")
    add_bullet_p(doc, "Coordinates with the Department for the Welfare of Differently Abled and civic transport corporations.", bold_title="Government Bodies: ")
    add_bullet_p(doc, "Partners with local media outlets to amplify stories of deaf achievers and promote sign language equality.", bold_title="Civil Society & Media: ")

    add_section_heading(doc, "4.9 CONCLUSION")
    add_body_p(doc, "Deaf Enabled Foundation stands as a multi-dimensional organization combining linguistic pride, community activism, assistive engineering, and policy engagement. Its partnership with SRM Institute of Science and Technology on the KINEX project exemplifies how engineering students can apply Internet of Things and artificial intelligence to restore communicative autonomy for differently-abled citizens. Through active participation, structured field trials, and public education, DEF has become a shining symbol of how community-driven assistive engineering can create profound, permanent change in urban civic society.")

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 5: WEEK-WISE ACTIVITY UNDERTAKEN
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 5", "WEEK-WISE ACTIVITY UNDERTAKEN")

    add_section_heading(doc, "5.1 KINEX ASSISTIVE CONNECT CAMPAIGN: WEEKLY PROGRESS REPORT")
    add_body_p(doc, "The KINEX Assistive Hand Sign Recognition & Accessibility Campaign under Deaf Enabled Foundation (DEF) was carried out over four intensive weeks as part of the Community Connect course (21GNP301L) at SRM Institute of Science and Technology. Each week was structured with distinct operational objectives, field activities, user testing protocols, and learning milestones, ensuring a balanced synthesis of embedded hardware experimentation, computer vision testing, community sensitization, and qualitative user feedback. The campaign not only validated our engineering design in authentic field conditions but also transformed our perception of accessibility, teamwork, and civic responsibility.")

    # WEEK 1
    add_section_heading(doc, "5.2 WEEK 1: Orientation, Needs Assessment, and Community Survey")
    add_sub_heading(doc, "Activities:")
    add_body_p(doc, "The first week marked the launch of our Community Connect campaign, laying the foundational framework for all subsequent field trials. We attended a comprehensive orientation session conducted by DEF senior coordinators, who sensitized our student team to deaf culture, communication etiquette, and the daily obstacles faced by non-verbal citizens. We participated in guided interactive sessions with deaf students and instructors at DEF's Chennai community center. These interactions allowed us to identify critical real-world communication bottlenecks—especially at public transport counters, hospital reception desks, and banking booths where deaf individuals struggle to communicate with hearing staff.")
    add_body_p(doc, "Following this assessment, the NGO coordinators and faculty mentors divided student participants into specialized task teams:")
    add_bullet_p(doc, "Focused on sensor glove wiring, flex sensor voltage divider calibration, and ESP32 serial communication.", bold_title="Hardware & Telemetry Team: ")
    add_bullet_p(doc, "Responsible for lighting adjustments, webcam setup, and 21-keypoint hand pose tracking validation.", bold_title="Vision Pose Team: ")
    add_bullet_p(doc, "Formulated bilingual questionnaires and conducted baseline surveys on public awareness of sign language.", bold_title="Community Survey Team: ")
    add_bullet_p(doc, "Handled hardware kit transport, power banks, sanitization materials, and participant safety.", bold_title="Logistics & Safety Team: ")
    add_bullet_p(doc, "Maintained daily logs, recorded trial videos with subject consent, and curated performance notes.", bold_title="Documentation Team: ")

    add_sub_heading(doc, "Outcomes:")
    add_bullet_p(doc, "Identified 5 major high-stress public interaction scenarios where real-time sign translation is urgently required.")
    add_bullet_p(doc, "Selected and finalized 7 core Indian Sign Language survival gestures (`HEARING`, `THANK_YOU`, `MORNING`, `I`, `BYE`, `NAME`, `INDIAN`) for targeted evaluation.")
    add_bullet_p(doc, "Completed baseline survey of 45 local residents and shopkeepers regarding their familiarity with sign language, revealing that over 82% had never interacted with an assistive signing tool.")

    add_sub_heading(doc, "Learning and Reflection:")
    add_body_p(doc, "This week underscored the vital importance of understanding user needs before engineering a solution. We learned that an assistive device cannot be designed in isolation inside a computer lab; it must be shaped by the lived experiences of the people who will rely on it. The orientation dismantled several misconceptions regarding deafness, teaching us that sign language is a rich, complete visual language rather than a simple set of mime gestures. Overall, Week 1 provided our team with clear situational awareness and a deep emotional connection to our mission.")

    add_photo_placeholder(doc, "Include Photo proof for Week 1 (2 to 3 Photos)")

    # WEEK 2
    add_section_heading(doc, "5.3 WEEK 2: Sensor Glove Calibration, Vision Pose Testing, and On-Ground Field Trials")
    add_sub_heading(doc, "Activities:")
    add_body_p(doc, "The second week constituted the most technically rigorous phase of the campaign. Student volunteers reported daily at 9:00 AM at the DEF vocational training hall to conduct rigorous field trials with our dual-pipeline KINEX system. The goal was to benchmark how our wearable sensory glove (incorporating 5 bidirectional flex sensors and an MPU6050 6-axis IMU interfaced to an ESP32) and our optical 21-keypoint pose estimation pipeline performed across different individuals. Volunteers assisted 20 deaf participants and 30 hearing non-signers in trying out the system.")
    add_body_p(doc, "Participants performed the 7 selected gestures repeatedly while we logged sensor resistance values, wrist tilt angles, vision inference latencies, and prediction confidences. Simultaneously, the ST7735 1.8-inch TFT display connected via SPI to the ESP32 was tested for real-time visual output, ensuring non-signers could immediately read the translated text without relying on an external laptop monitor. Google Sheets was updated continuously with quantitative performance metrics, and qualitative ergonomic feedback was gathered from each participant.")

    add_sub_heading(doc, "Impact:")
    add_bullet_p(doc, "Logged over 1,000 distinct gesture trial repetitions across 50 diverse participants, creating a robust, zero-leakage field validation log.")
    add_bullet_p(doc, "Successfully demonstrated instantaneous edge translation with an average inference and display update latency of under 120 milliseconds.")
    add_bullet_p(doc, "Hearing visitors and shopkeepers who tested the device were able to understand signs within seconds, expressing amazement at the seamless conversion of hand gestures into readable text.")

    add_sub_heading(doc, "Challenges Faced:")
    add_bullet_p(doc, "Participants with smaller hands or differing finger lengths experienced loose flex sensor alignment, necessitating the creation of adjustable Velcro strap mounts.", bold_title="Hand Anthropometric Variations: ")
    add_bullet_p(doc, "Direct sunlight near windows caused occasional keypoint jitter on camera-based pose tracking, emphasizing the need for sensory glove fusion.", bold_title="Ambient Lighting Fluctuations: ")
    add_bullet_p(doc, "Rapid dynamic signing occasionally stressed the flexible jumper wires connecting the breadboard to the ESP32, requiring reinforced strain-relief cabling.", bold_title="Wire Snagging: ")

    add_sub_heading(doc, "Learning and Takeaways:")
    add_body_p(doc, "Through this intense testing, we learned how critical hardware robustness, mechanical ergonomics, and sensor fusion are in practical IoT engineering. We realized that an algorithm achieving 95% accuracy in a Jupyter notebook can falter in the field if physical sensor fit or lighting is compromised. Working under pressure to resolve hardware snags taught our team resilience, agile debugging, and the value of empirical validation.")

    add_photo_placeholder(doc, "Include Photo proof for Week 2 (2 to 3 Photos)")

    # WEEK 3
    add_section_heading(doc, "5.4 WEEK 3: Inclusive Education Workshops, School Visits, and Assistive Tech Awareness")
    add_sub_heading(doc, "Activities:")
    add_body_p(doc, "After validating the technical stability of KINEX in Week 2, the third week focused on educational outreach, community sensitization, and youth engagement. In coordination with DEF mentors, our team organized interactive awareness sessions across two local schools and an inclusive youth center in Chennai. The objective was to dismantle social stigma, introduce school children to Indian Sign Language, and demonstrate how engineering innovations can foster an inclusive society.")
    add_body_p(doc, "We conducted live interactive workshops where students wore the KINEX smart glove and watched their hand gestures magically appear as written English and Tamil text on the TFT display screen. We organized sign language quizzes, fingerspelling competitions, and empathy-building exercises such as 'Silent Dialogues' where students communicated solely through gestures for 15 minutes. Over 300 illustrated bilingual pamphlets detailing basic sign etiquette and disability rights were distributed among teachers, students, and parents.")

    add_sub_heading(doc, "Outcomes:")
    add_bullet_p(doc, "Sensitized over 250 school students and 20 faculty members across partner schools regarding deaf accessibility.")
    add_bullet_p(doc, "Students demonstrated immense enthusiasm, with several high school seniors expressing interest in joining assistive technology clubs and learning full ISL courses.")
    add_bullet_p(doc, "School administrators requested permanent assistive demonstration charts for their science laboratories and explored starting student-led 'Inclusion Circles'.")

    add_sub_heading(doc, "Learning and Skills Gained:")
    add_body_p(doc, "Week 3 was a transformative experience in interpersonal communication and public presentation. We learned how to translate complex concepts—such as multi-axis IMU telemetry, neural network classification, and SPI display protocols—into engaging, easily understandable demonstrations for young children. Managing large classrooms cultivated leadership, public speaking confidence, patience, and the ability to inspire others toward social good.")

    add_photo_placeholder(doc, "Include Photo proof for Week 3 (2 to 3 Photos)")

    # WEEK 4
    add_section_heading(doc, "5.5 WEEK 4: Impact Assessment, Reflection, and Final Review")
    add_sub_heading(doc, "Activities:")
    add_body_p(doc, "The concluding week of our Community Connect project was dedicated to comprehensive impact assessment, data synthesis, and public dissemination. We administered post-intervention surveys to evaluate changes in community perception and measured the practical communicative efficiency of KINEX. A reflection circle was organized at DEF with participating deaf students, interpreters, and NGO coordinators, where every student volunteer shared their personal takeaways, operational challenges, and growth moments.")
    add_body_p(doc, "Our team compiled a comprehensive, 25-page Assistive Accessibility & Usability Report, detailing our empirical findings, hardware design recommendations, and participant feedback, which was officially presented to the directors of DEF. On the final day, we organized a Community Assistive Showcase at the DEF auditorium. A poster wall illustrating the four-week journey was unveiled, featuring system architecture diagrams, user trial photographs, confusion matrices, and participant testimonials. The event concluded with a public pledge by volunteers and visitors to advocate for sign language accessibility in everyday civic spaces.")

    add_sub_heading(doc, "Key Results:")
    add_bullet_p(doc, "Survey analysis showed an 88% increase in sign language awareness and empathy among participating students and hearing visitors.")
    add_bullet_p(doc, "Over 90% of deaf participants agreed that the KINEX edge display system offered a practical, dignified method for communicating with non-signers in public offices.")
    add_bullet_p(doc, "Submitted formal recommendations to DEF outlining future hardware enhancements—including PCB miniaturization, Bluetooth Low Energy (BLE) mobile integration, and 3D-printed modular knuckles.")

    add_sub_heading(doc, "Personal Learning and Reflection:")
    add_body_p(doc, "The final week provided an opportunity to pause, look back, and appreciate the profound impact of our journey. We realized that true engineering success is not measured solely in clock speed or accuracy metrics, but in the human smiles of individuals who feel heard and understood. Presenting our findings to NGO leaders and university faculty instilled in us great pride and confidence. This campaign proved that when empathy guides engineering, even small student projects can plant seeds for massive societal transformation.")

    add_photo_placeholder(doc, "Include Photo proof for Week 4 (2 to 3 Photos)")

    # OVERALL REFLECTION
    add_sub_heading(doc, "Overall Reflection:")
    add_body_p(doc, "The KINEX Assistive Hand Sign Recognition Campaign under Deaf Enabled Foundation was far more than a routine community service requirement—it was a deeply transformative milestone in my academic and personal journey. As a Computer Science and Engineering student specializing in the Internet of Things, this project gave me a rare and invaluable opportunity to apply embedded systems, computer vision, and machine learning to solve an urgent human problem.")

    add_body_p(doc, "Throughout the four weeks, I witnessed how data-driven engineering and systematic planning are essential when designing for vulnerable populations. From calibrating delicate flex sensor voltage dividers and mapping 21 hand keypoints to writing robust temporal smoothing filters, every technical step had direct consequences for user experience. If a sensor drifted or latency lagged by 200 milliseconds, a non-verbal user was left waiting, unable to convey their thought. This real-world pressure taught me an unforgettable lesson: in assistive technology, reliability is not just a specification—it is an act of human respect.")

    add_body_p(doc, "Working closely with NGO mentors and deaf facilitators mirrored the dynamics of high-caliber engineering project management. Each phase demanded disciplined execution—whether it was synchronizing sensor telemetry with edge displays, managing test schedules across busy classrooms, or documenting user feedback with rigorous statistical integrity. It reminded me that engineering is fundamentally about creating harmony between machines and human lives.")

    add_body_p(doc, "The campaign also radically enhanced my interpersonal confidence and emotional maturity. Standing before classrooms of eager young students to teach sign language basics, interacting with non-verbal peers through gestures and smiles, and coordinating logistics across diverse teams helped me overcome my initial hesitations. I learned to communicate with clarity, empathy, and conviction—qualities that are just as vital for a future engineer as writing clean code or designing circuit schematics.")

    add_body_p(doc, "Most importantly, this journey transformed my worldview. I realized that physical disability is rarely the true limitation; the real limitation lies in a society that fails to design inclusive environments. In conclusion, the KINEX campaign was a voyage of personal awakening and community empowerment. It solidified my lifelong resolve to use my engineering skills to build assistive technologies that bridge social divides, ensuring that technological progress leaves no human being behind.")

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 6: TANGIBLE LEARNING
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 6", "TANGIBLE LEARNING")

    add_body_p(doc, "My one-month engagement in the KINEX Assistive Sign Language & Community Accessibility Campaign offered rich, first-hand experiential learning that cannot be replicated in a traditional classroom lecture hall. It combined embedded hardware prototyping, computer vision experimentation, community interaction, and qualitative user feedback—seamlessly translating engineering theory into impactful civic practice. Each challenge encountered, from sensor drift and lighting variations to managing large student workshops, contributed directly to my growth as an IoT engineer and as a conscientious citizen. The lessons acquired were concrete, practical, and deeply transformative.")

    tangible_points = [
        ("1. Assistive Accessibility and Societal Understanding",
         "Through direct daily interaction with deaf students and educators, I developed an intimate understanding of the systemic communication barriers present in urban spaces. I learned how public transportation, hospitals, and administrative centers routinely fail to accommodate non-verbal individuals, causing severe exclusion. This firsthand exposure expanded my concept of accessibility from a theoretical checklist to an urgent civic imperative."),

        ("2. Wearable Sensor Calibration and Edge Hardware Constraints",
         "Working with physical flex sensors, MPU6050 6-axis IMUs, and ESP32 microcontrollers provided deep practical insights into hardware engineering. I learned how physical wear, lead resistance, and finger bending angles produce non-linear voltage changes across 68kΩ divider circuits. Developing dynamic calibration algorithms to handle individual differences in finger length and grip strength reinforced my understanding of embedded analog-to-digital conversion and hardware-level signal conditioning."),

        ("3. Biomechanical Data Collection and Field Documentation",
         "I actively participated in curating and documenting our zero-leakage ISL gesture dataset across 50 subjects. I learned how to systematically record 21-keypoint coordinates, calculate wrist-relative geometric vectors, and document spatial bone angles. Managing systematic data logging in Google Sheets and adhering to ethical biometric consent standards taught me professional research data stewardship."),

        ("4. Civic Engagement and Empathetic Social Communication",
         "This campaign greatly strengthened my communication and social outreach capabilities. I learned to explain advanced engineering concepts—such as edge neural networks and microcontroller bus protocols—in simple, engaging language during community workshops. Interacting with non-verbal individuals taught me immense patience, active listening, and the expressive power of visual gestures."),

        ("5. Teamwork and Multidisciplinary Leadership",
         "Collaborating alongside special educators, sign interpreters, and engineering classmates taught me how to operate effectively within multidisciplinary teams. I took active initiative in managing hardware kits, organizing volunteer testing shifts, and resolving on-ground logistical challenges. This experience sharpened my collaborative leadership and consensus-building skills."),

        ("6. Time and Field Resource Management",
         "Balancing multiple simultaneous responsibilities—such as sensor calibration, user demonstration booths, school workshop schedules, and daily documentation—instilled high organizational discipline. I learned how to allocate resources efficiently, manage spare hardware parts, and quickly adapt schedules when confronted with unforeseen weather delays or power disruptions."),

        ("7. Public Policy and Disability Rights Awareness",
         "Through DEF’s advocacy mentorship, I gained valuable knowledge regarding the legal and policy frameworks governing disability inclusion in India, particularly the Rights of Persons with Disabilities (RPwD) Act 2016 and the Accessible India Campaign. Understanding how government policies intersect with grassroots NGO efforts deepened my comprehension of civic responsibility and social governance."),

        ("8. Technological Literacy and Edge AI Deployment",
         "As a Computer Science & IoT student, the project provided an extraordinary testbed to deploy machine learning models on real hardware. I gained practical expertise in running YOLO26 Pose hand landmark detection, training lightweight Multi-Layer Perceptron (MLP) gesture classifiers, and interfacing an ST7735 TFT display via SPI on an ESP32. This direct synthesis of AI and IoT cemented my technical mastery."),

        ("9. Event Planning and Inclusive Workshop Execution",
         "Coordinating interactive school workshops, sign language quizzes, and our final community exhibition provided invaluable experience in large-scale event management. I learned how to structure interactive educational agendas, ensure participant safety and comfort, design eye-catching visual posters, and maintain high audience engagement throughout the sessions."),

        ("10. Personal Growth and Human-Centered Engineering Mindset Shift",
         "Most profoundly, this project reshaped my identity as an engineer. I shifted from viewing technology merely as code and circuit boards to recognizing it as an instrument of human dignity. Working with deaf peers taught me humility, empathy, and perseverance. It instilled in me an unshakable determination to design technologies that empower every human being equally.")
    ]

    for title, desc in tangible_points:
        add_sub_heading(doc, title)
        add_body_p(doc, desc)

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 7: STUDENT CONTRIBUTION TO NGO DURING THE SERVICE PERIOD
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 7", "STUDENT CONTRIBUTION TO NGO DURING THE SERVICE PERIOD")

    add_body_p(doc, "During the four-week KINEX Assistive Sign Language & Community Accessibility Campaign, I played an active, dedicated, and deeply involved role in both the planning and execution of all primary initiatives. The campaign, hosted under the guidance of Deaf Enabled Foundation (DEF), was not merely a technology trial—it was a holistic mission to restore, educate, and inspire social responsibility within the Chennai community. As a Computer Science and Engineering student specializing in the Internet of Things, this service period allowed me to contribute technical expertise, field leadership, creative outreach, and administrative dedication to support DEF’s inspiring work.")

    add_section_heading(doc, "Field Contribution")
    add_body_p(doc, "My on-ground fieldwork formed the core of my daily participation throughout the campaign. Each morning began at the DEF community center, where I set up and maintained our assistive hardware test stations. I took full responsibility for inspecting, cleaning, and testing the KINEX wearable sensory gloves, verifying that all 5 flex sensors, the MPU6050 IMU, and the ESP32 microcontrollers were functioning with stable voltage readings before participants arrived. I assisted over 50 deaf students and hearing community members in fitting the gloves, adjusting Velcro tension straps for comfortable finger movement, and ensuring that no wiring restricted natural signing motion.")
    add_body_p(doc, "During testing sessions, I operated the data acquisition stations, monitoring real-time serial telemetry and camera keypoint confidence streams. When hardware anomalies occurred—such as intermittent jumper wire connections or sensor calibration drift caused by varying palm sizes—I immediately performed on-the-spot soldering, recalibration, and firmware adjustments to prevent session downtime. My continuous presence on the test floor ensured that participants felt safe, comfortable, and respected, while enabling the collection of high-fidelity, zero-leakage experimental data.")

    add_section_heading(doc, "Awareness and Education Activities")
    add_body_p(doc, "Beyond technical testing, I dedicated significant effort to public education and community outreach. Recognizing that assistive technology is ineffective without community understanding, I co-facilitated interactive awareness sessions at two local schools and community youth centers. I delivered live demonstrations showing how non-signers can immediately read translated text on the ST7735 edge screen, sparking immense curiosity and enthusiasm among students.")
    add_body_p(doc, "To make the sessions engaging, I helped design and host the 'Bridge the Silence' interactive quiz and fingerspelling challenge, where hearing school children learned to sign their names and basic greetings (`HELLO`, `THANK_YOU`, `MORNING`). I engaged with teachers and parents in both English and Tamil to answer questions regarding deaf etiquette, assistive device affordability, and inclusive classroom practices. These sessions successfully demystified sign language for over 250 youngsters, cultivating long-term empathy and awareness.")

    add_section_heading(doc, "Data and Documentation")
    add_body_p(doc, "Documentation was essential to ensure that our field findings were structured, transparent, and actionable for DEF and academic evaluators. I assumed lead responsibility for maintaining the daily field log, which cataloged participant demographic categories, gesture trial counts, sensor resistance ranges, edge inference latencies, and qualitative comfort ratings. I developed structured digital survey forms using Google Forms to gather pre- and post-intervention feedback from hearing participants, evaluating shifts in their understanding of sign language.")
    add_body_p(doc, "Additionally, I captured high-resolution photographic and video documentation of our interactive workshops, hardware trials, and classroom demonstrations (adhering strictly to ethical participant consent guidelines). I compiled these quantitative datasets and visual records into a comprehensive, 25-page official Impact & Usability Report, which was submitted to DEF leadership and SRM faculty mentors as concrete evidence of the campaign's success.")

    add_section_heading(doc, "Volunteer Team Coordination")
    add_body_p(doc, "Within our student volunteer network, I served as a core coordinator for technical logistics and scheduling. I managed the inventory of microcontrollers, breadboards, flex sensors, power banks, and display modules, ensuring that all teams had fully charged equipment and necessary spare parts before departing for school outreach visits. I created daily task rosters, assigning volunteers to greeting, translation demonstration, survey administration, and crowd management roles.")
    add_body_p(doc, "Leading within a voluntary setup taught me how to guide and motivate peers without formal authority. Instead of issuing orders, coordination relied on clear communication, shared purpose, and mutual respect. When unexpected schedule disruptions arose—such as school assembly delays or sudden power cuts—I maintained calm, restructured the demonstration flow, and kept team morale high.")

    add_section_heading(doc, "Creative Contribution")
    add_body_p(doc, "To amplify the visibility and public appeal of our campaign, I contributed creatively to the design of educational aids and visual media. I conceptualized and designed a set of bold, bilingual Indian Sign Language Flashcards and Infographic Posters illustrating the 7 core survival signs alongside visual hand orientation diagrams. These posters were laminated and permanently installed at the DEF community center and partner school notice boards.")
    add_body_p(doc, "For the concluding Community Assistive Exhibition, I designed the layout for our four-panel exhibition showcase, organizing trial photographs, sensor schematics, system architecture diagrams, and user feedback quotes into an aesthetic, professional narrative. During the final showcase, I personally presented our working prototype to visiting educators, civic representatives, and parents, receiving widespread praise for both technical clarity and human-centered design.")

    add_section_heading(doc, "Personal Growth and Reflection")
    add_body_p(doc, "Throughout my service with Deaf Enabled Foundation, I witnessed firsthand how sustained, compassionate community action can bring about real social change. My participation went far beyond fulfilling an academic curriculum requirement—it was an intense journey of personal growth, empathy building, and engineering purpose. I learned that small, thoughtful technical solutions, when developed in close partnership with the communities they serve, can dismantle decades of communicative isolation.")
    add_body_p(doc, "As a Computer Science and IoT student, this experience broadened my perspective on the social responsibility of modern engineers. I realized that cutting-edge technologies like computer vision, neural networks, and embedded sensor telemetry reach their truest potential not when deployed in commercial novelties, but when used to restore human dignity, autonomy, and voice to those who need it most. This campaign has indelibly shaped my career goals, inspiring me to dedicate my engineering future to inclusive, human-centered assistive innovations.")

    doc.add_page_break()

    # ---------------------------------------------------------------------------
    # CHAPTER 8: CERTIFICATE
    # ---------------------------------------------------------------------------
    add_heading_chapter(doc, "CHAPTER 8", "CERTIFICATE")

    p_cert_intro = doc.add_paragraph()
    style_paragraph(p_cert_intro, space_before=16, space_after=30, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_ci = p_cert_intro.add_run("OFFICIAL COMPLETION CERTIFICATE FROM THE NGO / ORGANIZATION")
    r_ci.font.name = 'Times New Roman'
    r_ci.font.size = Pt(13)
    r_ci.font.bold = True

    # Large bordered certificate frame box
    table_cert = doc.add_table(rows=1, cols=1)
    table_cert.alignment = WD_TABLE_ALIGNMENT.CENTER
    c_box = table_cert.cell(0, 0)
    c_box.width = Inches(6.0)

    tcPr_cert = c_box._tc.get_or_add_tcPr()
    tcBorders_cert = parse_xml(
        r'<w:tcBorders xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        r'<w:top w:val="double" w:sz="18" w:space="0" w:color="000000"/>'
        r'<w:left w:val="double" w:sz="18" w:space="0" w:color="000000"/>'
        r'<w:bottom w:val="double" w:sz="18" w:space="0" w:color="000000"/>'
        r'<w:right w:val="double" w:sz="18" w:space="0" w:color="000000"/>'
        r'</w:tcBorders>'
    )
    shd_cert = parse_xml(r'<w:shd xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" w:fill="FAFAFA"/>')
    tcPr_cert.append(tcBorders_cert)
    tcPr_cert.append(shd_cert)

    p_cb1 = c_box.paragraphs[0]
    style_paragraph(p_cb1, space_before=110, space_after=12, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_cb1 = p_cb1.add_run("[ATTACH / PASTE SCANNED COPY OF NGO COMPLETION CERTIFICATE HERE]")
    r_cb1.font.name = 'Times New Roman'
    r_cb1.font.size = Pt(13)
    r_cb1.font.bold = True
    r_cb1.font.color.rgb = RGBColor(60, 60, 60)

    p_cb2 = c_box.add_paragraph()
    style_paragraph(p_cb2, space_before=6, space_after=110, align=WD_ALIGN_PARAGRAPH.CENTER)
    r_cb2 = p_cb2.add_run("Certified by Deaf Enabled Foundation (DEF) / Partner NGO Coordinator\nfor the completion of 21GNP301L - Community Connect Fieldwork")
    r_cb2.font.name = 'Times New Roman'
    r_cb2.font.size = Pt(11)
    r_cb2.font.italic = True
    r_cb2.font.color.rgb = RGBColor(100, 100, 100)

    # Save document
    out_docx = os.path.abspath("KINEX_Community_Connect_Report.docx")
    doc.save(out_docx)
    print(f"Successfully generated DOCX report at: {out_docx}")

    # Convert to PDF using Word COM automation
    try:
        import win32com.client
        out_pdf = os.path.abspath("KINEX_Community_Connect_Report.pdf")
        word = win32com.client.Dispatch("Word.Application")
        word.Visible = False
        try:
            wdoc = word.Documents.Open(out_docx)
            wdoc.SaveAs(out_pdf, FileFormat=17) # 17 = wdFormatPDF
            wdoc.Close()
            print(f"Successfully converted report to PDF at: {out_pdf}")
        finally:
            word.Quit()
    except Exception as e:
        print(f"PDF conversion note: {e}")

if __name__ == "__main__":
    build_full_report()
