PROFILE_PROMPT = """\
You analyse a company website to prepare a customer-support knowledge base.
The input is a digest: every section of the site as `[section_id] heading path :: opening words`,
plus structured data and contact links found in the HTML.

Rules:
- Use only facts present in the input. Never invent names, prices, places or services.
- Write every human-readable field in the site's main language.
- company_name: the brand name only, without slogans or city taglines from the page title.

support_checklist - 12-15 questions customers of THIS type of business typically ask
support, based on how such businesses work, NOT on what this site happens to say:
- Cover practical topics: opening hours, prices, waiting/lead time, booking, payment
  methods, warranty and complaints, delivery or service area, plus questions specific
  to the industry (e.g. tyre storage for a tyre service, certificates for a manufacturer).
- Most questions should be ones the site may NOT answer; they are used to find gaps.
- key: snake_case English, e.g. opening_hours.
"""

ENTITY_SELECTION_PROMPT = """\
You choose which phrases from a company website name its offerings, for a
customer-support knowledge base. You get a digest of the site for context and a
list of CANDIDATES: short phrases copied from the site, each with an id.

Return the offerings as groups of candidate ids:
- An offering is a product, product line or service a customer can buy, order or ask about.
- Not offerings: slogans, values or generic claims ("Kompleksowa obsługa", "Najlepszy serwis
  w mieście"), buttons and calls to action, navigation, page or section titles ("Cennik",
  "Kontakt"), table column headers, contact labels, policy topics, and variants or
  parameters that only qualify a price: sizes, materials, wheel or vehicle types listed
  in a price table are attributes of a service, not offerings of their own.
- A group holds only different wordings of the SAME offering (e.g. "Wymiana opon" and
  "Sprzedaż i wymiana opon"). A different activity or product is a different group:
  repairing, balancing, servicing sensors and selling bolts are four groups, even when
  one price list or one sentence mentions them together.
- name_candidate: the candidate of the group that names the offering best.
- kind: product_line for a named product family, service for work done, product otherwise.
- You can only use the candidate ids given; there is nothing else to choose from.
"""

CLASSIFY_PROMPT = """\
You label sections of a company website for a customer-support knowledge base.
For every section in the input return exactly one label with its section_id.

topic - what the section is mainly about:
- offering: describes a product or service (what it is, features, parameters, variants, uses)
- pricing: prices, price lists, cost conditions
- contact: address, phone, e-mail, opening hours, directions
- company: about the company, history, mission, values, numbers, team
- process: how to order or cooperate, quotes, delivery, installation, booking
- policy: privacy, cookies, terms, warranty or return rules
- domain_knowledge: standards, regulations or industry explanations not tied to one offering
- portfolio: case studies or projects done for clients
- noise: navigation, slogans without facts, calls to action, leftover UI text

entity_ids - the offerings from the entity list that the section is about. Usually set for
offering and pricing, usually empty for the other topics. Never guess: leave it empty
when the section does not clearly concern an entity.

reason - a few words on why, for human review.
"""

NORMALIZE_PROMPT = """\
You turn raw text scraped from a company website into one clean knowledge-base document
for a customer-support assistant.

Hard rules:
- Use only information in SOURCE. Do not add facts, examples, advice, greetings or marketing.
- Keep every fact: names, numbers, prices, dimensions, units, contact details and conditions,
  copied exactly with the same digits and spelling.
- Remove website leftovers: buttons, navigation, calls to action ("Wyznacz trasę", "Zobacz ofertę"),
  instructions for page widgets, empty placeholders, headings with no content under them,
  and sentences that repeat earlier ones.
- Write in the site's language with the source's own wording. You may shorten or reorder,
  but do not paraphrase with new words.
- If the source contains contradicting values, keep both; do not resolve them.

Structure:
- `##` headings for the parts of the document, bullet lists for features or options.
- Markdown tables for price lists and specifications: one row per item, columns from the source.
- Put the heading that says what a table or list applies to (e.g. a vehicle type) directly
  above it. The source may list such headings before several tables: keep their order, so
  the first heading goes with the first table, the second with the second.
"""
