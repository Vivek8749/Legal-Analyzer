"""Risk taxonomy: the kinds of one-sided terms and loopholes the analyzer looks for.

Each category carries retrieval queries (used to pull the relevant clauses out of the
document) and a description (given to Claude as the review brief). Categories are grouped
into review passes; each pass is one Claude request.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Group:
    id: str
    name: str
    description: str


@dataclass(frozen=True)
class Category:
    id: str
    name: str
    group: str
    description: str
    queries: tuple[str, ...]


GROUPS: tuple[Group, ...] = (
    Group("changes", "Changes, renewal & termination",
          "Who can change, renew, or end the agreement, and on what terms."),
    Group("liability", "Liability, warranties & risk",
          "How losses and legal risk are divided between the parties."),
    Group("disputes", "Disputes & legal rights",
          "How disagreements are resolved and which legal rights are given up."),
    Group("money", "Fees, payments & penalties",
          "What you pay, when prices can change, and what happens if you pay late or leave."),
    Group("data_ip", "Data, content & intellectual property",
          "What happens to your data, content, ideas, and confidential information."),
    Group("obligations", "Obligations, restrictions & control",
          "Duties and restrictions placed on you, and who controls the relationship."),
    Group("drafting", "Drafting traps & ambiguity",
          "Vague wording, linked documents, and boilerplate that quietly shifts power."),
)

CATEGORIES: tuple[Category, ...] = (
    # --- Changes, renewal & termination -------------------------------------------------
    Category(
        "unilateral_changes", "Unilateral changes", "changes",
        "One party (usually the drafter) can change the terms, prices, features, or policies on its own: "
        "at any time, with little or no notice, or by treating continued use as acceptance. Check whether "
        "the other party gets advance notice, a right to reject, or a right to leave with a refund.",
        ("we may modify amend or update these terms at any time",
         "changes become effective when posted and continued use means you accept the revised terms",
         "we reserve the right to change modify or discontinue the service features or prices without notice",
         "notice of material changes to the agreement"),
    ),
    Category(
        "termination_suspension", "Termination & suspension", "changes",
        "Rights to suspend or end the agreement: termination 'for any reason' or without notice or a chance "
        "to fix a problem, termination rights only one side has, and harsh consequences of termination such "
        "as losing data, prepaid fees, or accrued rights.",
        ("we may suspend or terminate your account at any time for any reason or no reason without notice",
         "termination for convenience upon written notice",
         "effect of termination refund data deletion loss of access",
         "you may terminate this agreement only by",
         "breach cure period before termination"),
    ),
    Category(
        "auto_renewal", "Auto-renewal & cancellation", "changes",
        "Automatic renewal or rollover, cancellation windows that close long before renewal, cancellation or "
        "early-termination fees, minimum terms, and burdensome cancellation procedures.",
        ("automatically renews for successive renewal terms unless cancelled",
         "notice of non-renewal must be received days before the end of the term",
         "early termination fee cancellation charge minimum term",
         "how to cancel your subscription"),
    ),
    # --- Liability, warranties & risk ---------------------------------------------------
    Category(
        "limitation_of_liability", "Limitation of liability", "liability",
        "Caps on damages and exclusions of liability: very low caps, exclusion of indirect or consequential "
        "losses, exclusions that cover the drafter's own negligence or misconduct, and caps that protect only "
        "one party.",
        ("in no event shall we be liable for any indirect incidental special consequential or punitive damages",
         "total aggregate liability shall not exceed the amount you paid",
         "limitation of liability exclusion of damages",
         "liability for loss of data profits or business interruption"),
    ),
    Category(
        "indemnification", "Indemnification", "liability",
        "Duties to indemnify (pay for) or defend the other party against claims: one-sided or uncapped "
        "indemnities, broad triggers such as 'arising out of or related to your use', covering the other "
        "party's own negligence, and one party controlling the defence at the other's cost.",
        ("you agree to indemnify defend and hold harmless us and our affiliates from any claims",
         "indemnification obligations losses damages costs and attorneys fees",
         "arising out of or relating to your use of the service or breach of these terms",
         "we will indemnify you"),
    ),
    Category(
        "warranty_disclaimer", "Warranty disclaimers", "liability",
        "Disclaimers that the product or service comes 'as is', with no promise of quality, accuracy, "
        "availability, security, or fitness for purpose, and service-level promises hollowed out by exclusions.",
        ("the service is provided as is and as available without warranties of any kind",
         "we disclaim all implied warranties of merchantability fitness for a particular purpose and non-infringement",
         "we do not guarantee uptime availability accuracy or that the service will be error free",
         "service level availability credits exclusions scheduled maintenance"),
    ),
    Category(
        "force_majeure", "Force majeure", "liability",
        "Events that excuse a party from performing: definitions broad enough to include ordinary business "
        "problems, protection for only one party, or payment duties that continue while the other side is excused.",
        ("force majeure events beyond our reasonable control",
         "not liable for delay or failure to perform due to acts of god strikes or outages",
         "payment obligations are not excused"),
    ),
    # --- Disputes & legal rights --------------------------------------------------------
    Category(
        "arbitration_class_waiver", "Arbitration & class-action waiver", "disputes",
        "Mandatory binding arbitration, waivers of class actions or jury trial, short or burdensome opt-out "
        "procedures, arbitrator or venue chosen by one party, cost-shifting, and carve-outs that let only one "
        "party go to court.",
        ("binding individual arbitration instead of court",
         "class action waiver you waive the right to participate in a class action",
         "waiver of jury trial",
         "opt out of arbitration within days by sending written notice",
         "small claims court exception injunctive relief intellectual property claims"),
    ),
    Category(
        "governing_law_venue", "Governing law & venue", "disputes",
        "Which law applies and where disputes must be brought: distant or foreign courts, exclusive "
        "jurisdiction chosen by the drafter, and clauses that make enforcing rights expensive for the other party.",
        ("this agreement shall be governed by the laws of",
         "exclusive jurisdiction and venue of the courts located in",
         "you consent to personal jurisdiction"),
    ),
    Category(
        "claims_rights_waivers", "Time limits & waived rights", "disputes",
        "Shortened deadlines to bring claims, waivers or releases of legal rights and remedies, waivers of "
        "statutory or consumer protections, and limits on the remedies available.",
        ("any claim must be filed within one year after the cause of action arose or be forever barred",
         "you waive any right to",
         "you release us from all claims",
         "your sole and exclusive remedy is"),
    ),
    # --- Fees, payments & penalties -----------------------------------------------------
    Category(
        "fees_and_price_changes", "Fees, refunds & price changes", "money",
        "Fees and charges: price increases without consent, hidden or vaguely defined fees, non-refundable "
        "payments, refund restrictions, automatic charges, and taxes or currency costs pushed onto one party.",
        ("fees are non-refundable",
         "we may change our prices at any time",
         "you authorize us to charge your payment method automatically",
         "additional fees administrative charges and taxes",
         "refund policy"),
    ),
    Category(
        "penalties_late_fees", "Penalties & late fees", "money",
        "Late fees, high interest, liquidated damages, collection costs, acceleration of payments, deposit "
        "forfeiture, and other penalties, especially when only one party can be penalised.",
        ("late payment fee and interest per month on overdue amounts",
         "liquidated damages penalty",
         "collection costs and attorneys fees",
         "suspend service for non-payment",
         "security deposit forfeited"),
    ),
    # --- Data, content & intellectual property ------------------------------------------
    Category(
        "data_privacy", "Data collection & sharing", "data_ip",
        "Use and sharing of personal data: broad collection, sharing or selling to affiliates and 'partners', "
        "vague purposes, indefinite retention, consent to tracking or marketing, and privacy terms that can "
        "change unilaterally.",
        ("we may share your personal information with affiliates partners and third parties",
         "we collect use and retain your data",
         "consent to receive marketing communications",
         "data retention and deletion after termination",
         "privacy policy incorporated by reference"),
    ),
    Category(
        "content_ip_license", "Your content & IP", "data_ip",
        "Rights taken over the other party's content, work, or ideas: perpetual, irrevocable, worldwide, "
        "royalty-free or sublicensable licences; IP assignment or work-for-hire; waiver of moral rights; free "
        "use of feedback; and use of name or likeness.",
        ("you grant us a perpetual irrevocable worldwide royalty free sublicensable license to use your content",
         "all intellectual property created shall be owned by the company as work made for hire",
         "feedback and suggestions may be used without compensation",
         "you waive moral rights",
         "use of your name and likeness"),
    ),
    Category(
        "confidentiality", "Confidentiality", "data_ip",
        "Confidentiality duties: obligations on only one side, overly broad definitions of confidential "
        "information, never-ending duties, or restrictions that stop a party from complaining or reporting.",
        ("confidential information shall not be disclosed",
         "obligations of confidentiality survive termination",
         "non-disparagement"),
    ),
    # --- Obligations, restrictions & control --------------------------------------------
    Category(
        "restrictive_covenants", "Non-compete & restrictions", "obligations",
        "Restrictions on future activity: non-compete, non-solicitation, exclusivity, and non-disparagement "
        "clauses, especially when broad in time, geography, or scope.",
        ("non-compete shall not engage in any competing business",
         "non-solicitation of employees or customers",
         "exclusivity you agree not to use competing services",
         "restricted period after termination"),
    ),
    Category(
        "assignment_control", "Assignment & transfer", "obligations",
        "Who can transfer the agreement: one party may assign or transfer it (including data) freely, for "
        "example in a merger, while the other cannot; subcontracting without consent.",
        ("we may assign or transfer this agreement without your consent",
         "you may not assign or transfer your rights",
         "change of control merger or acquisition"),
    ),
    Category(
        "obligations_notices", "Your obligations & notices", "obligations",
        "Burdensome or open-ended duties placed on one party (compliance, monitoring, audits, acceptable-use "
        "rules judged at the other's discretion) and lopsided notice rules, such as one side notifying by "
        "posting online while the other must use certified mail.",
        ("you are responsible for all activities under your account",
         "acceptable use policy prohibited conduct at our sole discretion",
         "notices to you may be given by posting on the website or by email",
         "notices to us must be sent in writing by certified mail",
         "audit and inspect records"),
    ),
    # --- Drafting traps & ambiguity -----------------------------------------------------
    Category(
        "vague_discretion", "Vague & discretionary language", "drafting",
        "Wording that lets one party decide later: 'sole discretion', 'reasonable', 'material', 'from time to "
        "time', 'including without limitation', undefined key terms, and asymmetries where one side 'may' while "
        "the other 'shall'.",
        ("at our sole and absolute discretion",
         "as we deem appropriate from time to time",
         "including without limitation",
         "material change as determined by us",
         "commercially reasonable efforts"),
    ),
    Category(
        "incorporated_documents", "Linked & hidden documents", "drafting",
        "Terms pulled in by reference from other documents, policies, or web pages (often changeable at will), "
        "and order-of-precedence clauses that let those documents override the main agreement.",
        ("incorporated herein by reference",
         "policies and guidelines posted on our website as amended from time to time",
         "in the event of a conflict the following order of precedence applies"),
    ),
    Category(
        "boilerplate_traps", "Boilerplate traps", "drafting",
        "Standard-looking clauses with real effects: entire-agreement clauses that cancel sales promises, "
        "survival of one-sided terms after termination, no-waiver and severability wording that preserves the "
        "drafter's rights, electronic consent, and deemed acceptance.",
        ("entire agreement supersedes all prior representations and statements",
         "provisions that survive termination or expiration",
         "no waiver failure to enforce any right",
         "severability unenforceable provision shall be modified",
         "by clicking accept or using the service you agree"),
    ),
)

# Queries that find the parts of a document describing what it is and who the parties are.
OVERVIEW_QUERIES: tuple[str, ...] = (
    "this agreement is entered into between the parties",
    "definitions of terms used in this agreement",
    "effective date and term of the agreement",
    "governing law of this agreement",
)

CATEGORY_IDS: tuple[str, ...] = tuple(c.id for c in CATEGORIES)
CATEGORY_BY_ID: dict[str, Category] = {c.id: c for c in CATEGORIES}
GROUP_BY_ID: dict[str, Group] = {g.id: g for g in GROUPS}


def categories_in(group_id: str) -> list[Category]:
    return [c for c in CATEGORIES if c.group == group_id]
