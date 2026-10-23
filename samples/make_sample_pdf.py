"""Generate samples/sample_terms_of_service.pdf, a deliberately one-sided Terms of Service.

The company is fictional. The document mixes obvious problems (a $10 liability cap, a 7-day
arbitration opt-out) with subtler loopholes (an uptime promise whose exclusions swallow it,
a data-export right that immediate termination makes illusory) so you can see what the
analyzer catches. Requires the dev extra: pip install -e ".[dev]"
"""

from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

TITLE = "TERMS OF SERVICE"
PREAMBLE = [
    "Acme Cloud Storage, Inc. - Last updated: March 1, 2026",
    "(Sample document for testing. Acme Cloud Storage, Inc. is a fictional company.)",
    'Please read these Terms of Service ("Terms") carefully. By creating an account, clicking "I agree", or '
    'accessing or using the Acme Cloud Storage service (the "Service"), you agree to be bound by these Terms. '
    "If you do not agree, do not use the Service.",
]

SECTIONS: list[tuple[str, list[str]]] = [
    ("1. DEFINITIONS", [
        '1.1 "Acme", "we", "us" and "our" mean Acme Cloud Storage, Inc., a Delaware corporation, and its affiliates.',
        '1.2 "You" and "your" mean the individual or entity that registers for or uses the Service.',
        '1.3 "Your Content" means any files, data, text, images, or other materials you upload, store, or share '
        "through the Service.",
        '1.4 "Subscription Term" means the initial period of your paid subscription and each renewal period.',
        '1.5 "Material Change" means a change that Acme, in its sole discretion, determines to be material.',
    ]),
    ("2. CHANGES TO THESE TERMS", [
        "2.1 We may revise these Terms, our fees, or any part of the Service at any time and for any reason. "
        "Revised Terms become effective immediately when posted on our website. Your continued use of the Service "
        "after the revised Terms are posted constitutes your acceptance of them.",
        "2.2 We will provide 30 days' notice of any Material Change by posting a notice on our website. Changes "
        "that are not Material Changes may be made without notice. It is your responsibility to review these Terms "
        "periodically.",
    ]),
    ("3. ACCOUNTS AND ELIGIBILITY", [
        "3.1 You must be at least 18 years old to use the Service. You are responsible for maintaining the "
        "confidentiality of your password and for all activities that occur under your account, whether or not "
        "you authorized them.",
        "3.2 You agree to provide accurate information and to keep it up to date. Acme may refuse service to "
        "anyone for any reason at any time.",
    ]),
    ("4. SUBSCRIPTIONS, FEES AND AUTOMATIC RENEWAL", [
        "4.1 Paid plans are billed in advance on a monthly or annual basis. Your subscription will automatically "
        "renew for successive renewal periods equal in length to the initial Subscription Term unless you cancel "
        "at least sixty (60) days before the end of the then-current Subscription Term.",
        "4.2 You authorize Acme to charge your payment method automatically for each renewal period at the "
        "then-current rates, which Acme may change at any time. Acme may also charge reasonable administrative "
        "fees, which will be described on our website from time to time.",
        "4.3 All fees are non-refundable, including for partial subscription periods, unused storage, downgrades, "
        "and periods during which the Service was unavailable, except as required by law.",
        "4.4 Late payments bear interest at the rate of 3% per month, or the maximum rate permitted by law, "
        "whichever is higher. You will reimburse Acme for all costs of collection, including attorneys' fees.",
        "4.5 If you cancel before the end of an annual Subscription Term, you will pay an early termination fee "
        "equal to the remaining fees for that Subscription Term.",
    ]),
    ("5. YOUR CONTENT", [
        "5.1 You retain ownership of Your Content. However, by uploading Your Content, you grant Acme and its "
        "affiliates a perpetual, irrevocable, worldwide, royalty-free, sublicensable and transferable license to "
        "host, copy, use, modify, distribute, publicly display, and create derivative works from Your Content for "
        "any purpose, including to develop and train new products and services, and including after you stop "
        "using the Service.",
        "5.2 You waive any moral rights or rights of attribution you may have in Your Content. Any feedback or "
        "suggestions you provide may be used by Acme without restriction or compensation to you.",
        "5.3 You represent and warrant that you own or have all rights necessary to upload Your Content and to "
        "grant the license above.",
    ]),
    ("6. PRIVACY AND DATA", [
        "6.1 Our collection and use of personal information is described in our Privacy Policy, which is "
        "incorporated into these Terms by reference and which we may update from time to time without notice to you.",
        "6.2 You agree that Acme may share information about you and your use of the Service, including the "
        "metadata of Your Content, with its affiliates, advertising partners, and other trusted third parties for "
        "any business purpose.",
        "6.3 Acme may retain your information and Your Content for as long as it determines is necessary for its "
        "business purposes, including after your account is closed.",
        "6.4 You consent to receive marketing emails, text messages, and telephone calls from Acme and its partners.",
    ]),
    ("7. SECURITY AND AVAILABILITY", [
        "7.1 Acme will use commercially reasonable efforts to keep the Service available 99.9% of the time in each "
        'calendar month (the "Uptime Commitment"). Scheduled maintenance, emergency maintenance, and any '
        "unavailability caused by third-party providers are excluded from the uptime calculation. Acme may perform "
        "scheduled or emergency maintenance at any time, with or without notice.",
        "7.2 If Acme fails to meet the Uptime Commitment, your sole and exclusive remedy is a service credit equal "
        "to 5% of the monthly fee for the affected month, which must be requested in writing within five (5) days "
        "after the end of that month. Service credits have no cash value and expire if not used within 30 days.",
        "7.3 You are solely responsible for backing up Your Content. Acme is not responsible for any loss or "
        "corruption of Your Content.",
    ]),
    ("8. ACCEPTABLE USE", [
        "You will not use the Service to store or share content that is unlawful, infringing, harmful, or "
        "otherwise objectionable, as determined by Acme in its sole discretion. Acme may monitor, review, and "
        "remove any of Your Content at any time without notice. You agree to cooperate with any investigation and "
        "to provide Acme with any information it requests.",
    ]),
    ("9. SUSPENSION AND TERMINATION", [
        "9.1 Acme may suspend or terminate your account or access to the Service, in whole or in part, at any "
        "time, for any reason or no reason, with or without notice.",
        "9.2 You may terminate your account only by sending written notice by certified mail to Acme's legal "
        "department at least thirty (30) days in advance. Cancellation through the website or by email is not "
        "effective.",
        "9.3 Upon any termination, your right to use the Service ends immediately and Acme may delete Your Content "
        "immediately and without liability. You may request an export of Your Content before termination takes "
        "effect; Acme will process such requests within a reasonable time.",
        "9.4 No fees will be refunded upon termination for any reason, including termination by Acme without cause.",
    ]),
    ("10. DISCLAIMER OF WARRANTIES", [
        'THE SERVICE IS PROVIDED "AS IS" AND "AS AVAILABLE" WITHOUT WARRANTIES OF ANY KIND, WHETHER EXPRESS, '
        "IMPLIED, OR STATUTORY. ACME DISCLAIMS ALL IMPLIED WARRANTIES, INCLUDING MERCHANTABILITY, FITNESS FOR A "
        "PARTICULAR PURPOSE, TITLE, AND NON-INFRINGEMENT. ACME DOES NOT WARRANT THAT THE SERVICE WILL BE "
        "UNINTERRUPTED, SECURE, OR ERROR-FREE, OR THAT YOUR CONTENT WILL NOT BE LOST.",
    ]),
    ("11. LIMITATION OF LIABILITY", [
        "11.1 TO THE MAXIMUM EXTENT PERMITTED BY LAW, IN NO EVENT WILL ACME OR ITS AFFILIATES BE LIABLE FOR ANY "
        "INDIRECT, INCIDENTAL, SPECIAL, CONSEQUENTIAL, EXEMPLARY, OR PUNITIVE DAMAGES, OR FOR ANY LOSS OF DATA, "
        "PROFITS, REVENUE, OR BUSINESS, EVEN IF ACME HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGES AND EVEN "
        "IF CAUSED BY ACME'S OWN NEGLIGENCE.",
        "11.2 ACME'S TOTAL LIABILITY FOR ALL CLAIMS ARISING OUT OF OR RELATING TO THESE TERMS OR THE SERVICE WILL "
        "NOT EXCEED THE GREATER OF TEN U.S. DOLLARS (US $10) OR THE FEES YOU PAID TO ACME IN THE ONE (1) MONTH "
        "BEFORE THE CLAIM AROSE.",
        "11.3 The limitations in this Section 11 do not apply to your obligations under Section 12 or to any "
        "amounts you owe to Acme.",
    ]),
    ("12. INDEMNIFICATION", [
        "You agree to defend, indemnify, and hold harmless Acme, its affiliates, and their officers, directors, "
        "employees, and agents from and against any and all claims, damages, losses, liabilities, costs, and "
        "expenses (including attorneys' fees) arising out of or relating to your use of the Service, Your Content, "
        "or your breach of these Terms, including claims caused in part by Acme's negligence. Acme may assume "
        "exclusive control of the defense of any claim, at your expense, and you may not settle any claim without "
        "Acme's prior written consent.",
    ]),
    ("13. DISPUTE RESOLUTION AND ARBITRATION", [
        "13.1 Any dispute arising out of or relating to these Terms or the Service will be resolved by final and "
        "binding individual arbitration administered by an arbitration provider selected by Acme, and not in "
        "court. The arbitration will take place in Wilmington, Delaware.",
        "13.2 YOU AND ACME EACH WAIVE THE RIGHT TO A JURY TRIAL AND THE RIGHT TO PARTICIPATE IN A CLASS ACTION, "
        "CLASS ARBITRATION, OR REPRESENTATIVE PROCEEDING.",
        "13.3 You may opt out of this arbitration agreement by mailing a signed written notice to Acme's legal "
        "department within seven (7) days after you first accept these Terms.",
        "13.4 Notwithstanding the foregoing, Acme may bring claims to protect its intellectual property, to "
        "collect unpaid fees, or to seek injunctive relief in any court of competent jurisdiction.",
        "13.5 Each party will bear its own costs of arbitration, except that you will reimburse Acme's attorneys' "
        "fees if Acme prevails.",
    ]),
    ("14. GOVERNING LAW AND VENUE", [
        "These Terms are governed by the laws of the State of Delaware, without regard to its conflict-of-law "
        "rules. Subject to Section 13, you consent to the exclusive jurisdiction of the state and federal courts "
        "located in Wilmington, Delaware.",
    ]),
    ("15. TIME LIMIT ON CLAIMS", [
        "Any claim you may have arising out of or relating to these Terms or the Service must be filed within six "
        "(6) months after the claim arose; otherwise, the claim is permanently barred.",
    ]),
    ("16. ASSIGNMENT", [
        "Acme may assign or transfer these Terms, and any of its rights and obligations, including Your Content "
        "and personal information, to any person without your consent, including in connection with a merger, "
        "acquisition, or sale of assets. You may not assign or transfer these Terms without Acme's prior written "
        "consent.",
    ]),
    ("17. FORCE MAJEURE", [
        "Acme will not be liable for any delay or failure to perform caused by events beyond its reasonable "
        "control, including acts of God, labor shortages, supplier failures, cyberattacks, power or internet "
        "outages, or changes in law. Your obligation to pay fees is not excused by any such event.",
    ]),
    ("18. NOTICES", [
        "Acme may give you notice by email, by posting on the Service, or by posting on our website; notice is "
        "effective when sent or posted. Notices to Acme must be in writing, sent by certified mail to Acme Cloud "
        "Storage, Inc., Attn: Legal Department, 100 Example Plaza, Wilmington, DE 19801, and are effective only "
        "when received.",
    ]),
    ("19. GENERAL", [
        "19.1 Entire Agreement. These Terms, together with the Privacy Policy and any other policies Acme posts on "
        "its website, are the entire agreement between you and Acme and supersede all prior statements, "
        "advertisements, and representations, written or oral.",
        "19.2 Severability. If any provision is found unenforceable, it will be modified to the minimum extent "
        "necessary to make it enforceable, and the remaining provisions remain in full effect.",
        "19.3 No Waiver. Acme's failure to enforce any right or provision is not a waiver of that right or provision.",
        "19.4 Survival. Sections 4, 5, 6, 9.3, 9.4, 10, 11, 12, 13, 15 and 19, and any other provisions that by "
        "their nature should survive, survive any termination or expiration of these Terms.",
        "19.5 Electronic Communications. You consent to receive all agreements, notices, and disclosures "
        "electronically.",
    ]),
    ("20. CONTACT", [
        "Questions about these Terms may be sent to legal@acme-cloud.example.",
    ]),
]


class _TermsPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 8, "Acme Cloud Storage, Inc. - Terms of Service", align="R")
        self.ln(10)

    def footer(self) -> None:
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()} of {{nb}}", align="C")


def build(path: Path) -> Path:
    pdf = _TermsPDF(format="Letter")
    pdf.set_title("Acme Cloud Storage Terms of Service (sample)")
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, TITLE, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 10)
    for paragraph in PREAMBLE:
        pdf.multi_cell(0, 5, paragraph, align="J", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
    for heading, paragraphs in SECTIONS:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, heading, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        for paragraph in paragraphs:
            pdf.multi_cell(0, 5, paragraph, align="J", new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1.5)
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))
    return path


if __name__ == "__main__":
    out = build(Path(__file__).with_name("sample_terms_of_service.pdf"))
    print(f"Wrote {out}")
