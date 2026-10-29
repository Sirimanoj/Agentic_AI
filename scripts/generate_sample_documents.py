"""Generate high-quality realistic sample legal documents for demonstration and benchmarking."""

import os
from pathlib import Path
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def create_sample_documents(output_dir: str = "data/sample_documents") -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Generate sample_nda.pdf (3 pages)
    pdf_path = out_path / "sample_nda.pdf"
    c = canvas.Canvas(str(pdf_path), pagesize=letter)

    # Page 1
    t = c.beginText(54, 750)
    t.setFont("Helvetica-Bold", 14)
    t.textLine("MUTUAL NON-DISCLOSURE AND CONFIDENTIALITY AGREEMENT")
    t.setFont("Helvetica", 10)
    t.textLine("")
    t.textLine("This Non-Disclosure Agreement (the \"Agreement\") is entered into as of January 15, 2026,")
    t.textLine("by and between Apex Technologies Corp. (\"Disclosing Party\") and Novus Legal AI Inc. (\"Receiving Party\").")
    t.textLine("")
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 1: DEFINITIONS OF CONFIDENTIAL INFORMATION")
    t.setFont("Helvetica", 10)
    t.textLine("\"Confidential Information\" refers to any proprietary, technical, financial, or commercial data,")
    t.textLine("software code, trade secrets, business plans, customer records, and product architectures disclosed")
    t.textLine("by Disclosing Party to Receiving Party, whether orally, in writing, or through electronic systems.")
    t.textLine("")
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 2: OBLIGATIONS AND DUTY OF CARE")
    t.setFont("Helvetica", 10)
    t.textLine("The Receiving Party shall exercise a reasonable degree of care, and in no event less than standard")
    t.textLine("industry practices, to safeguard Confidential Information against unauthorized copying or dissemination.")
    t.textLine("Receiving Party shall strictly restrict disclosure to employees and contractors with a bona fide need to know.")
    c.drawText(t)
    c.showPage()

    # Page 2
    t = c.beginText(54, 750)
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 3: PERMITTED DISCLOSURES AND LEGAL COMPULSION")
    t.setFont("Helvetica", 10)
    t.textLine("The Receiving Party may disclose Confidential Information to the limited extent required by valid law,")
    t.textLine("subpoena, civil investigative demand, or binding order of a court of competent jurisdiction.")
    t.textLine("Provided, however, that Receiving Party shall provide prompt advance written notice within 5 business days")
    t.textLine("to Disclosing Party to enable Disclosing Party to seek a protective order or confidential treatment.")
    t.textLine("")
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 4: TERM AND SURVIVAL OF CONFIDENTIALITY")
    t.setFont("Helvetica", 10)
    t.textLine("This Agreement shall remain in effect for a period of two (2) years from the Effective Date.")
    t.textLine("Notwithstanding termination or expiration of this Agreement, the confidentiality obligations with")
    t.textLine("respect to all disclosed Confidential Information shall survive for a period of three (3) years")
    t.textLine("following the effective date of termination. Trade secrets shall be maintained in confidence in perpetuity.")
    c.drawText(t)
    c.showPage()

    # Page 3
    t = c.beginText(54, 750)
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 5: RETURN AND CERTIFICATION OF DESTRUCTION")
    t.setFont("Helvetica", 10)
    t.textLine("Within fourteen (14) calendar days following Disclosing Party's written demand, Receiving Party shall")
    t.textLine("promptly return or securely destroy all tangible materials, digital documents, summaries, and notes")
    t.textLine("containing or reflecting Confidential Information, and provide an executive written certificate of destruction.")
    t.textLine("")
    t.setFont("Helvetica-Bold", 11)
    t.textLine("SECTION 6: GOVERNING LAW AND FORUM")
    t.setFont("Helvetica", 10)
    t.textLine("This Agreement shall be governed by, and construed in accordance with, the substantive laws of the")
    t.textLine("State of Delaware, without regard to its conflict of laws principles.")
    t.textLine("The parties irrevocably submit to the exclusive jurisdiction of the Court of Chancery of the State of Delaware.")
    c.drawText(t)
    c.showPage()
    c.save()
    print(f"Generated PDF: {pdf_path}")

    # 2. Generate sample_employment_agreement.docx
    docx_path = out_path / "sample_employment_agreement.docx"
    doc = Document()
    doc.add_heading("EXECUTIVE EMPLOYMENT AGREEMENT", level=0)

    doc.add_heading("SECTION 1: APPOINTMENT AND SCOPE OF DUTIES", level=1)
    doc.add_paragraph(
        "Company hereby engages Employee to serve in the position of Chief Technology Officer (CTO). "
        "Employee shall report directly to the Chief Executive Officer and Board of Directors. "
        "Employee agrees to devote Employee's entire business time, attention, skill, and best efforts "
        "to the business and affairs of the Company."
    )

    doc.add_heading("SECTION 2: COMPENSATION, SALARY, AND INCENTIVES", level=1)
    doc.add_paragraph(
        "Employee shall receive an annualized base salary of $220,000 USD, payable in accordance with the "
        "Company's standard payroll schedule (semi-monthly). In addition, Employee is eligible for an annual "
        "discretionary cash performance bonus with a target amount equal to 25% of annual base salary, "
        "predicated on corporate milestones approved by the Compensation Committee."
    )

    doc.add_heading("SECTION 3: TERMINATION AND NOTICE PERIOD", level=1)
    doc.add_paragraph(
        "Employment is at-will and may be terminated by either party at any time. However, either the Company "
        "or Employee must provide not less than thirty (30) days prior written notice of termination without Cause. "
        "In the event the Company terminates Employee's employment without Cause, Company shall pay Employee "
        "severance compensation equal to six (6) months of base salary subject to execution of a standard release."
    )

    doc.add_heading("SECTION 4: NON-COMPETITION AND NON-SOLICITATION", level=1)
    doc.add_paragraph(
        "During employment and for a restricted duration of twelve (12) months following separation of employment, "
        "Employee shall not directly or indirectly engage in, manage, or advise any competing enterprise operating "
        "in the legal AI and automated document processing sector within North America. Furthermore, Employee shall not "
        "solicit any company client, vendor, or employee for twelve (12) months following termination."
    )

    doc.add_heading("SECTION 5: INTELLECTUAL PROPERTY AND INVENTIONS ASSIGNMENT", level=1)
    doc.add_paragraph(
        "All discoveries, software code, algorithms, inventions, patents, and copyrightable works created or "
        "conceived by Employee during the term of employment that relate directly to the Company's business "
        "shall be deemed 'works made for hire' and remain the sole and exclusive proprietary property of the Company."
    )

    doc.save(docx_path)
    print(f"Generated DOCX: {docx_path}")

    # 3. Generate sample_statute_excerpts.txt
    txt_path = out_path / "sample_statute_excerpts.txt"
    statute_text = (
        "UNIFORM COMMERCIAL CODE (UCC) & CONTRACT LAW STATUTORY EXCERPTS\n\n"
        "§ 2-201. FORMAL REQUIREMENTS; STATUTE OF FRAUDS.\n"
        "(1) Except as otherwise provided in this section a contract for the sale of goods for the price of $500 or more "
        "is not enforceable by way of action or defense unless there is some writing sufficient to indicate that a "
        "contract for sale has been made between the parties and signed by the party against whom enforcement is sought "
        "or by his authorized agent or broker. A writing is not insufficient because it omits or incorrectly states a "
        "term agreed upon but the contract is not enforceable under this paragraph beyond the quantity of goods shown in such writing.\n"
        "(2) Between merchants if within a reasonable time a writing in confirmation of the contract and sufficient against "
        "the sender is received and the party receiving it has reason to know its contents, it satisfies the requirements "
        "of subsection (1) against such party unless written notice of objection to its contents is given within 10 days after it is received.\n\n"
        "§ 2-718. LIQUIDATION OR LIMITATION OF DAMAGES; DEPOSITS.\n"
        "(1) Damages for breach by either party may be liquidated in the agreement but only at an amount which is "
        "reasonable in the light of the anticipated or actual harm caused by the breach, the difficulties of proof of loss, "
        "and the inconvenience or nonfeasibility of otherwise obtaining an adequate remedy. A term fixing unreasonably "
        "large liquidated damages is void as a penalty.\n"
        "(2) Where the seller justifiably withholds delivery of goods because of the buyer's breach, the buyer is entitled "
        "to restitution of any amount by which the sum of his payments exceeds twenty percent of the value of the total performance "
        "for which the buyer is obligated under the contract or $500, whichever is smaller.\n\n"
        "§ 2-719. CONTRACTUAL MODIFICATION OR LIMITATION OF REMEDY.\n"
        "(1) Subject to the provisions of subsections (2) and (3) of this section and of the preceding section on liquidation and limitation of damages,\n"
        "(a) the agreement may provide for remedies in addition to or in substitution for those provided in this Article and may limit or alter "
        "the measure of damages recoverable under this Article, as by limiting the buyer's remedies to return of the goods and repayment of "
        "the price or to repair and replacement of non-conforming goods or parts; and\n"
        "(b) resort to a remedy as provided is optional unless the remedy is expressly agreed to be exclusive, in which case it is the sole remedy.\n"
        "(2) Where circumstances cause an exclusive or limited remedy to fail of its essential purpose, remedy may be had as provided in this Act.\n"
        "(3) Consequential damages may be limited or excluded unless the limitation or exclusion is unconscionable. Limitation of consequential "
        "damages for injury to the person in the case of consumer goods is prima facie unconscionable but limitation of damages where the loss "
        "is commercial is not.\n"
    )

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(statute_text)
    print(f"Generated TXT: {txt_path}")


if __name__ == "__main__":
    create_sample_documents()

