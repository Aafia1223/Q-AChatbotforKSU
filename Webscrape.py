import asyncio
import aiohttp
import json
import hashlib
import os
import time
from datetime import datetime
from urllib.parse import urljoin
from bs4 import BeautifulSoup
import re
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options



OUTPUT_DIR = "data_backups"
OUTPUT_FILE = "hierarchy.json"
BASE = "https://ksu.edu.sa"
UA = {"User-Agent": "Mozilla/5.0"}

is_arabic = False

# Ensure output directory exists
def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"📁 Created directory: {OUTPUT_DIR}")

def save_to_json(data, filename=None):
    """Save data to JSON file with proper formatting"""
    ensure_output_dir()
    
    if filename is None:
        filename = OUTPUT_FILE
    
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Data successfully saved to: {filepath}")
        print(f"📊 File size: {os.path.getsize(filepath)} bytes")
        
        # Optional: Create a backup with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_filename = f"menu_hierarchy_backup_{timestamp}.json"
        backup_filepath = os.path.join(OUTPUT_DIR, backup_filename)
        
        with open(backup_filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"🔄 Backup created: {backup_filepath}")
        
    except Exception as e:
        print(f"❌ Error saving to JSON: {e}")
        raise

# Session management for connection pooling
async def create_session():
    connector = aiohttp.TCPConnector(limit=50, limit_per_host=20)
    timeout = aiohttp.ClientTimeout(total=30)
    return aiohttp.ClientSession(
        connector=connector,
        timeout=timeout,
        headers=UA
    )

async def get_soup(session, path_or_url):
    global is_arabic

    url = path_or_url if path_or_url.startswith("http") else urljoin(BASE, path_or_url)
    
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            text = await response.text()
            soup = BeautifulSoup(text, "html.parser")

            if not is_arabic:  # Only check for Arabic and redirect if needed
                html_tag = soup.find("html")
                if html_tag and html_tag.get("lang", "").startswith("ar"):
                    eng_link = soup.find("a", string=lambda text: text and "English" in text)
                    if eng_link and eng_link.get("href"):
                        eng_url = urljoin(url, eng_link["href"])
                        print(f"🔁 Switching to English version: {eng_url}")
                        async with session.get(eng_url) as eng_response:
                            eng_response.raise_for_status()
                            eng_text = await eng_response.text()
                            soup = BeautifulSoup(eng_text, "html.parser")
                            is_arabic = True

            return soup
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        raise

def remove_arabic(text):
    # Arabic Unicode block: \u0600 to \u06FF, plus extended blocks if needed
    arabic_re = re.compile(r'[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+')
    return arabic_re.sub('', text)

async def find_faculty_links(session, dep_url):
    try:
        soup = await get_soup(session, dep_url)
    except Exception:
        return []

    links = []
    for a in soup.select("a[href]"):
        txt = remove_arabic(a.get_text(strip=True)).lower()
        if any(kw in txt for kw in ["faculty", "staff", "member", "academic team", "people", "employee", "employees"]):
            href = a["href"]
            if href:
                full_url = urljoin(dep_url, href)
                if full_url.startswith("http"):
                    links.append({
                        "title": a.get_text(strip=True),
                        "url": full_url
                    })
    return links

def find_contact_link(soup, base_url):
    for a in soup.select("a[href]"):
        text = remove_arabic(a.get_text(strip=True)).lower()
        if "contact" in text:
            href = a["href"]
            return urljoin(base_url, href)
    return None

async def extract_contact_info(session, contact_url):
    """Scrape the Contact Us page and extract phone, email, location, and other info."""
    try:
        soup = await get_soup(session, contact_url)
    except Exception:
        return "Contact page could not be loaded."

    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.split("\n") if line.strip()]

    phone = "Not found"
    email = "Not found"
    location = "Not found"
    extras = []

    for line in lines:
        lowered = line.lower()

        if email == "Not found" and re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", line):
            email = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", line).group(0)
            continue

        if phone == "Not found" and re.search(r"(\+?\d[\d\s\-()]{7,})", line):
            phone = re.search(r"(\+?\d[\d\s\-()]{7,})", line).group(0)
            continue

        if location == "Not found" and any(x in lowered for x in ["ksa", "riyadh", "building", "street", "road", "kingdom", "campus", "p.o", "box", "hall"]):
            location = line
            continue

        if any(x in lowered for x in ["fax", "hours", "box", "p.o", "linkedin", "facebook", "twitter", "instagram", "ext", "extension", "mobile", "call"]):
            extras.append(line)

    extra_text = "\n".join(extras) if extras else "None"
    return f"Phone Number: {phone}, Email: {email}, Location: {location}\nOthers:\n{extra_text}"

async def fetch_about_of(session, url):
    """
    Fetch the "about" text from a department (or college) URL,
    ignoring any <p> that contains an <img>.
    """
    try:
        soup = await get_soup(session, url)
    except Exception:
        return ""

    # 1) Drupal body block
    block = soup.select_one("div.field--name-body")
    if block:
        paras = []
        for p in block.find_all("p"):
            if p.find("img"):
                continue
            text = remove_arabic(p.get_text(" ", strip=True))
            if text:
                paras.append(text)
        if paras:
            return " ".join(paras)

    # 2) Views‑body fallback
    span_block = soup.select_one(
        "span.views-field.views-field-body span.field-content"
    )
    if span_block:
        paras = []
        for p in span_block.find_all("p"):
            if p.find("img"):
                continue
            text = remove_arabic(p.get_text(" ", strip=True))
            if text:
                paras.append(text)
        if paras:
            return " ".join(paras)

    # 3) Heading‑driven
    hdr = soup.find(
        lambda t: t.name in ("h2","h3","h4")
        and any(kw in t.text for kw in ("About Department","About the Department","About College"))
    )
    if hdr:
        collected = []
        for elem in hdr.find_all_next():
            if elem.name in ("h2","h3","h4"):
                break
            if elem.name == "p" and not elem.find("img"):
                txt = elem.get_text(" ", strip=True)
                if txt:
                    collected.append(txt)
        if collected:
            return " ".join(collected)

    # 4) Ultimate fallback: first pure <p> starting "The …"
    for p in soup.find_all("p"):
        if p.find("img"):
            continue
        txt = p.get_text(" ", strip=True)
        if txt.startswith("The"):
            return txt

    return ""

def find_main_menu_ul(soup):
    for nav in soup.find_all("nav"):
        ul = nav.find("ul")
        if ul and len(ul.find_all("li", recursive=False)) >= 5:
            return ul
    header = soup.find("header") or soup
    best, bc = None, 0
    for ul in header.find_all("ul"):
        cnt = len(ul.find_all("li", recursive=False))
        if cnt > bc:
            best, bc = ul, cnt
    if best and bc >= 5:
        return best
    raise RuntimeError("Main menu <ul> not found")

def recurse_menu(ul):
    out = []
    for li in ul.find_all("li", recursive=False):
        a = li.find("a", recursive=False)
        title = a.get_text(strip=True) if a else li.get_text(strip=True)
        href  = a["href"] if a and a.has_attr("href") else None
        sub = li.find("ul", recursive=False)
        out.append({
            "title":    title,
            "url":      href,
            "children": recurse_menu(sub) if sub else []
        })
    return out

async def scrape_menu(session):
    soup = await get_soup(session, "/en/home")
    main_ul = find_main_menu_ul(soup)
    return recurse_menu(main_ul)

async def scrape_category(session, path):
    soup = await get_soup(session, path)
    view = soup.find("div", class_="view-content")
    items = []
    if view:
        for card in view.find_all("div", class_="views-row"):
            a = card.find("a", href=True)
            if not a:
                continue
            items.append({
                "title":    a.get_text(strip=True),
                "url":      a["href"],
                "children": []
            })
    return items

def get_departments(soup, base_url):
    """Try to extract department links either from academic section or fallback menu."""
    departments = []

    # Primary: find "Academic Departments" or similar header
    hdr = soup.find(lambda t: t.name in ["h2", "h3", "h4"] and any(
        kw in t.get_text(strip=True).lower() for kw in ["academic departments", "departments", "academic"]
    ))

    if hdr:
        ul = hdr.find_next_sibling(lambda t: t.name == "ul")
        if ul:
            for li in ul.find_all("li"):
                a = li.find("a", href=True)
                if a:
                    departments.append({
                        "title": a.get_text(strip=True),
                        "url": urljoin(base_url, a["href"])
                    })

    # Fallback: try sidebars, navs, menus
    if not departments:
        menus = soup.find_all(["nav", "aside", "div"], class_=lambda c: c and "menu" in c.lower())
        for menu in menus:
            for a in menu.find_all("a", href=True):
                text = a.get_text(strip=True).lower()
                if "department" in text:
                    departments.append({
                        "title": a.get_text(strip=True),
                        "url": urljoin(base_url, a["href"])
                    })

    return departments

def extract_js_departments(college_url):
    """Uses Selenium to extract department links from JS-rendered sections like carousels."""
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--log-level=3")  # Reduce console logs

    driver = webdriver.Chrome(options=options)
    driver.get(college_url)
    time.sleep(3)  # Wait for JS to load

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    driver.quit()

    departments = []

    # Look for known carousel sections, or any link that looks like a department
    for a in soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        href = a["href"]
        if any(kw in text for kw in ["department", "unit", "program", "academic"]) and href != "#":
            full_url = href if href.startswith("http") else urljoin(college_url, href)
            departments.append({
                "title": a.get_text(strip=True),
                "url": full_url
            })

    return departments

async def scrape_dar_faqs(session):
    url = "https://dar.ksu.edu.sa/en/faqs"
    soup = await get_soup(session, url)

    faq_list = []
    content_block = soup.select_one("div.region-content") or soup.select_one("main") or soup.select_one("div.main-content")

    if not content_block:
        return []

    current_question = None

    for tag in content_block.find_all(["strong", "p", "div"]):
        if tag.name == "strong":
            q_text = tag.get_text(strip=True)
            if q_text:
                current_question = q_text
        elif current_question:
            a_text = tag.get_text(strip=True)
            if a_text:
                faq_list.append({
                    "question": current_question,
                    "answer": a_text
                })
                current_question = None

    return faq_list

def deduplicate_faqs(faqs):
    seen = set()
    deduped = []
    for faq in faqs:
        q = faq["question"]
        if q not in seen:
            deduped.append(faq)
            seen.add(q)
    return deduped

def scrape_faqs_with_selenium():
    base_url = "https://www.ksu.edu.sa/en"

    # Step 1: Launch headless browser
    options = Options()
    options.headless = True
    driver = webdriver.Chrome(options=options)
    driver.get(base_url)

    # Step 2: Get the FAQs link from the footer
    faq_link = None
    time.sleep(3)  # Let JS load
    soup = BeautifulSoup(driver.page_source, "html.parser")
    footer = soup.find("footer")
    if footer:
        for a in footer.find_all("a", href=True):
            if "faq" in a["href"].lower():
                faq_link = urljoin(base_url, a["href"])
                break

    if not faq_link:
        print("❌ Could not find FAQ link in footer.")
        driver.quit()
        return []

    # Step 3: Visit FAQs page
    print(f"🔗 Visiting FAQ page: {faq_link}")
    driver.get(faq_link)
    time.sleep(3)
    fsoup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()

    # Step 4: Scrape FAQ content
    faq_items = fsoup.select("div.faq-item")
    if not faq_items:
        print("❌ No FAQ items found after rendering.")
        return []

    faq_children = []
    for item in faq_items:
        summary = item.find("summary")
        answer_div = item.find("div", class_="faq-body")

        if not summary or not answer_div:
            continue

        question = summary.get_text(strip=True)
        answer = answer_div.get_text(separator=" ", strip=True)

        faq_children.append({
            "title": question,
            "answer": answer
        })

    return [{
        "title": "FAQs",
        "url": faq_link,
        "children": faq_children
    }]

async def merge_all_faqs(session):
    # Scrape both sources concurrently
    ksu_task = asyncio.create_task(asyncio.to_thread(scrape_faqs_with_selenium))
    dar_task = asyncio.create_task(scrape_dar_faqs(session))
    
    ksu_faq_sections, dar_faq_items = await asyncio.gather(ksu_task, dar_task)

    # Ensure we have base KSU FAQs section
    if not ksu_faq_sections:
        print("❌ Could not load KSU FAQs.")
        return []

    # Extract the first (and only) section
    ksu_faq_section = ksu_faq_sections[0]

    # Convert DAR items into same structure as KSU's children
    dar_children = [{
        "title": item["question"],
        "answer": item["answer"],
        "source": "https://dar.ksu.edu.sa/en/faqs"
    } for item in dar_faq_items]

    # Optional: Deduplicate all FAQs (by question text)
    combined_children = ksu_faq_section["children"] + dar_children
    deduped_children = deduplicate_faqs([
        {"question": c["title"], "answer": c["answer"]} for c in combined_children
    ])

    # Re-wrap into children format
    final_children = [{
        "title": item["question"],
        "answer": item["answer"]
    } for item in deduped_children]

    # Update and return unified section
    return [{
        "title": "FAQs",
        "url": ksu_faq_section["url"],
        "children": final_children
    }]

async def scrape_plagiarism_content(session):
    url = "https://chss.ksu.edu.sa/en/plagiarism-en"
    soup = await get_soup(session, url)
    content_block = soup.select_one("div.region-content") or soup.select_one("main") or soup.select_one("div.main-content")

    if not content_block:
        return ""

    text_elements = content_block.find_all(["p", "h2", "h3", "li"])
    content = "\n".join(p.get_text(strip=True) for p in text_elements if p.get_text(strip=True))
    return content

async def scrape_grading_system_table(session):
    url = "https://dar.ksu.edu.sa/en/node/815"
    soup = await get_soup(session, url)
    
    # Find the first table in the content region
    table = soup.find("table")
    if not table:
        return ""

    rows = []
    for tr in table.find_all("tr"):
        cols = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
        rows.append(" | ".join(cols))

    # Format as markdown-style table
    if not rows:
        return ""

    header = rows[0]
    separator = " | ".join(["---"] * len(header.split(" | ")))
    body = "\n".join(rows[1:])
    return f"**KSU Grading System Table:**\n\n{header}\n{separator}\n{body}"

async def scrape_regulations_and_policies(session):
    print("📘 Scraping Regulations and Policies section...")
    base_url = "https://ksu.edu.sa/en/policies"
    soup = await get_soup(session, base_url)

    # Use the main content area (not header or footer)
    content_block = soup.select_one("div.region-content") or soup.select_one("main") or soup.select_one("div.main-content")

    # Extract base page content
    page_content = ""
    if content_block:
        main_text_elements = content_block.find_all(["p", "h2", "h3", "li"])
        page_content = "\n".join(p.get_text(strip=True) for p in main_text_elements if p.get_text(strip=True))

    # Extract child links in main content only
    children = []
    seen_urls = set()

    if content_block:
        for a in content_block.select("a[href]"):
            title = a.get_text(strip=True)
            href = a["href"].strip()

            if not title or "javascript" in href.lower() or href.startswith("#"):
                continue

            full_url = urljoin(base_url, href)
            if full_url in seen_urls:
                continue
            seen_urls.add(full_url)

            children.append({
                "title": title,
                "url": full_url
            })

    # Get plagiarism and grading content concurrently
    plagiarism_task = asyncio.create_task(scrape_plagiarism_content(session))
    grading_task = asyncio.create_task(scrape_grading_system_table(session))
    
    plagiarism_content, grading_content = await asyncio.gather(plagiarism_task, grading_task)

    # ✅ Manually add extra children (outside the loop)
    children.append({
        "title": "Plagiarism",
        "url": "https://chss.ksu.edu.sa/en/plagiarism-en",
        "content": plagiarism_content
    })

    children.append({
        "title": "Grading System",
        "url": "https://dar.ksu.edu.sa/en/node/815",
        "content": grading_content,
        "info": "https://engineering.ksu.edu.sa/sites/engineering.ksu.edu.sa/files/imce_images/regulations_of_study_and_examinations_of_ksu.pdf"
    })

    return {
        "title": "Regulations and Policies",
        "url": base_url,
        "content": page_content,
        "children": children
    }

async def scrape_admission_requirements(session):
    print("📘 Scraping Admission Requirements section...")
    url = "https://graduatestudies.ksu.edu.sa/en/node/859"
    soup = await get_soup(session, url)

    table = soup.find("table")
    children = []

    if table:
        rows = table.find_all("tr")
        for row in rows:
            cols = row.find_all(["td", "th"])
            if len(cols) >= 2:
                title = cols[0].get_text(strip=True)
                content = cols[1].get_text(separator="\n", strip=True)
                if title and content:
                    children.append({
                        "title": title,
                        "url": url,
                        "content": content
                    })

    return {
        "title": "Admission Requirements",
        "url": url,
        "content": "Graduate admission criteria as listed by the Deanship of Graduate Studies.",
        "children": children
    }

async def scrape_research_institutes(session):
    url = "https://ksu.edu.sa/en/node/3106"
    soup = await get_soup(session, url)
    content_block = soup.select_one("div.region-content") or soup.select_one("main") or soup.select_one("div.main-content")
    if not content_block:
        return []

    institutes = []
    text = content_block.get_text(separator="\n").strip()
    # the page lists institute names in plain text—best-effort split lines:
    for line in text.split("\n"):
        name = line.strip()
        if name and not name.lower().startswith("do you like"):
            institutes.append({"title": name})
    return institutes

async def scrape_library_section(session):
    base_url = "https://library.ksu.edu.sa"
    page_url = f"{base_url}/en"
    
    async with session.get(page_url) as response:
        response.raise_for_status()
        text = await response.text()
        soup = BeautifulSoup(text, "html.parser")

    # Arabic → English mapping
    title_map = {
        "مكتبات مشتركة": "Shared libraries",
        "Men libraries": "Men libraries",
        "مكتبات الطالبات": "Female libraries"
    }

    # Find "المكتبات" tab
    menu_items = soup.select("li.menu-item--expanded > a")
    libraries_link = None
    for a in menu_items:
        if "المكتبات" in a.get_text(strip=True):
            libraries_link = a.find_parent("li")
            break

    if not libraries_link:
        print("❌ Could not find المكتبات menu.")
        return None

    libraries_section = {
        "title": "Libraries",
        "url": page_url,
        "children": []
    }

    # Process each library type concurrently
    async def process_library_type(a):
        arabic_title = a.get_text(strip=True)
        url = a["href"]
        full_url = urljoin(base_url, url)
        english_title = title_map.get(arabic_title, arabic_title)

        # Scrape children links inside the page's main content
        async with session.get(full_url) as response:
            response.raise_for_status()
            page_text = await response.text()
            page_soup = BeautifulSoup(page_text, "html.parser")
            
        main_content = (
            page_soup.select_one("main") or
            page_soup.select_one("div.region-content") or
            page_soup.select_one("div.main-content") or
            page_soup.body
        )

        link_children = []
        if main_content:
            # Process child pages concurrently
            async def process_child_link(link):
                text = link.get_text(strip=True)
                href = link["href"]

                if not text or href.startswith("#") or "mailto:" in href:
                    return None

                child_url = urljoin(full_url, href)

                try:
                    async with session.get(child_url) as child_resp:
                        child_resp.raise_for_status()
                        child_text = await child_resp.text()
                        child_soup = BeautifulSoup(child_text, "html.parser")
                        
                    child_main = (
                        child_soup.select_one("main") or
                        child_soup.select_one("div.region-content") or
                        child_soup.select_one("div.main-content") or
                        child_soup.body
                    )

                    # Extract paragraphs as "Information"
                    paragraphs = [p.get_text(separator="\n", strip=True) for p in child_main.find_all("p")]
                    info_section = {
                        "title": "Information",
                        "content": "\n\n".join(paragraphs)
                    } if paragraphs else None

                    # Extract tables as "Contact info" - filter out empty tables
                    tables = []
                    for table in child_main.find_all("table"):
                        rows = []
                        for row in table.find_all("tr"):
                            cols = [cell.get_text(strip=True) for cell in row.find_all(["td", "th"])]
                            # Only add row if it has non-empty content
                            if cols and any(col.strip() for col in cols):
                                rows.append(cols)

                        headers = rows[0] if rows and table.find("th") else []
                        data_rows = rows[1:] if headers else rows

                        # Only add table if it has meaningful content
                        if rows and any(any(cell.strip() for cell in row) for row in rows):
                            tables.append({
                                "headers": headers,
                                "rows": data_rows
                            })

                    contact_info_section = {
                        "title": "Contact info",
                        "tables": tables
                    } if tables else None

                    # Extract location link
                    location_section = None
                    for a_tag in child_main.find_all("a", href=True):
                        if "click here" in a_tag.get_text(strip=True).lower():
                            location_section = {
                                "title": "Location",
                                "url": urljoin(child_url, a_tag["href"])
                            }
                            break

                    child_sections = []
                    if info_section:
                        child_sections.append(info_section)
                    if contact_info_section:
                        child_sections.append(contact_info_section)
                    if location_section:
                        child_sections.append(location_section)

                    return {
                        "title": text,
                        "url": child_url,
                        "children": child_sections
                    }

                except Exception as e:
                    return None

            # Process all child links concurrently
            child_tasks = [process_child_link(link) for link in main_content.find_all("a", href=True)]
            child_results = await asyncio.gather(*child_tasks, return_exceptions=True)
            
            # Filter out None results and exceptions
            link_children = [result for result in child_results if result is not None and not isinstance(result, Exception)]

        return {
            "title": english_title,
            "url": full_url,
            "children": link_children
        }

    # Process all library types concurrently
    library_tasks = [process_library_type(a) for a in libraries_link.select("ul.menu a")]
    library_results = await asyncio.gather(*library_tasks, return_exceptions=True)
    
    # Filter out exceptions
    libraries_section["children"] = [result for result in library_results if not isinstance(result, Exception)]

    return libraries_section

async def scrape_academic_calendar(session):
    url = "https://dar.ksu.edu.sa/en/CurrentCalendar"
    
    async with session.get(url) as response:
        response.raise_for_status()
        text = await response.text()
        soup = BeautifulSoup(text, "html.parser")

    # Attempt to locate a <table> first
    tbl = soup.find("table")
    if tbl:
        headers = [th.get_text(strip=True) for th in tbl.select("thead th")] if tbl.find("thead") else []
        rows = [
            [cell.get_text(strip=True) for cell in row.find_all(["td","th"])]
            for row in tbl.find_all("tr")
        ]
        table_content = {
            "headers": headers,
            "rows": rows
        }
    else:
        # Fallback: parse free‑text to key‑value rows
        text_content = soup.get_text(separator="\n")
        lines = [ln.strip() for ln in text_content.split("\n") if ln.strip()]
        # Remove footer notices
        lines = [ln for ln in lines if not ln.lower().startswith("last updated")]
        table_content = {
            "text_rows": lines
        }

    return {
        "title": "Academic Calendar",
        "url": url,
        "table": table_content
    }

async def scrape_housing_section(session):
    housing_url = f"{BASE}/en/housing"
    soup = await get_soup(session, housing_url)

    async def extract_links_from_tab(tab_id):
        tab_div = soup.select_one(tab_id)
        if not tab_div:
            return []

        children = []
        for link in tab_div.find_all("a", href=True):
            title = link.get_text(strip=True)
            href = link["href"]
            if not title or href.startswith("#") or "mailto:" in href:
                continue

            full_url = urljoin(BASE, href)
            children.append({
                "title": title,
                "url": full_url,
                "children": []
            })
        return children

    # Extract faculty and student links concurrently
    faculty_task = extract_links_from_tab("#nav-faculty")
    student_task = extract_links_from_tab("#nav-students")
    
    faculty_links, student_links = await asyncio.gather(faculty_task, student_task)

    # Add Procedural Guide for Registration in Student Housing
    procedural_url = "https://sa.ksu.edu.sa/en/node/1013"
    procedural_soup = await get_soup(session, procedural_url)
    article = procedural_soup.select_one("article")
    procedural_text = article.get_text(separator="\n", strip=True) if article else ""

    student_links.append({
        "title": "Procedural Guide for Registration in Student Housing",
        "url": procedural_url,
        "content": procedural_text,
        "children": []
    })

    student_base_url = "https://sa.ksu.edu.sa/en/node/1007"
    student_soup = await get_soup(session, student_base_url)

    # Find the parent <li> that links to /en/node/6649
    female_menu_li = student_soup.select_one('li.menu-item--expanded > a[href="/en/node/6649"]')
    if female_menu_li:
        parent_li = female_menu_li.find_parent("li")
        submenu = parent_li.find("ul", class_="menu sub-menu")

        female_children = []

        if submenu:
            for a in submenu.find_all("a", href=True):
                title = a.get_text(strip=True)
                full_url = urljoin(student_base_url, a["href"])

                female_children.append({
                    "title": title,
                    "url": full_url,
                    "children": []
                })

        student_links.append({
            "title": "Registration in Female Student Housing",
            "url": urljoin(student_base_url, "/en/node/6649"),
            "children": female_children
        })

    # Scrape "RELATED LINKS" from the English housing site
    faculty_housing_url = "https://housing.ksu.edu.sa/en/"
    faculty_soup = await get_soup(session, faculty_housing_url)

    related_links_url = ""
    related_children = []

    # Find <a> tag that says "Related links" (case-insensitive)
    for a in faculty_soup.find_all("a", href=True):
        text = a.get_text(strip=True).lower()
        if "related links" in text:
            related_links_url = urljoin(faculty_housing_url, a["href"])
            print(f"[✅] Found Related Links URL: {related_links_url}")
            break

    # Now fetch the Related Links page if found
    if related_links_url:
        related_links_section = await get_soup(session, related_links_url)

        content_area = related_links_section.select_one("article")

        if content_area:
            links = []
            for p in content_area.find_all("p"):
                a = p.find("a", href=True)
                if not a:
                    continue
                href = a["href"]
                strong = a.find("strong")
                title = strong.get_text(strip=True) if strong else a.get_text(strip=True)
                if not title or href.startswith("#") or "mailto:" in href:
                    continue
                full_url = urljoin(related_links_url, href)
                links.append({
                    "title": title,
                    "url": full_url,
                })

            # Deduplicate links by (title + url)
            seen = set()
            deduped_links = []
            for link in links:
                key = (link["title"], link["url"])
                if key not in seen:
                    deduped_links.append(link)
                    seen.add(key)

            if deduped_links:
                related_children.append({
                    "title": "Related Links",
                    "url": related_links_url,
                    "children": deduped_links
                })

        # Add "Related Links" as a sub-section under Faculty Housing
        faculty_links.extend(related_children)

    return {
        "title": "Housing",
        "url": housing_url,
        "children": [
            {
                "title": "Faculty Housing",
                "children": faculty_links
            },
            {
                "title": "Student Housing",
                "children": student_links
            }
        ]
    }

def build_it_helpdesk_tree(csv_path):
    df = pd.read_csv(csv_path)

    # Clean and drop invalid rows
    df.dropna(subset=["Audience", "Category", "Subcategory"], inplace=True)

    # Normalize strings
    df["Audience"] = df["Audience"].str.strip().str.title()
    df["Category"] = df["Category"].str.strip().str.title()
    df["Subcategory"] = df["Subcategory"].str.strip().str.title()

    # Remove exact duplicate rows
    df.drop_duplicates(subset=["Audience", "Category", "Subcategory"], inplace=True)

    # Build tree structure: Audience -> Category -> Subcategory
    audiences = {}
    for _, row in df.iterrows():
        aud = row["Audience"]
        cat = row["Category"]
        sub = row["Subcategory"]

        if aud not in audiences:
            audiences[aud] = {}

        if cat not in audiences[aud]:
            audiences[aud][cat] = set()

        audiences[aud][cat].add(sub)

    # Construct final JSON
    audience_nodes = []
    for aud, cats in audiences.items():
        category_nodes = []
        for cat, subcats in cats.items():
            sub_nodes = [{"title": s, "children": []} for s in sorted(subcats)]
            category_nodes.append({
                "title": f"Categories: {cat}",
                "children": sub_nodes
            })
        audience_nodes.append({
            "title": aud,
            "children": category_nodes
        })

    tree = {
        "title": "IT Helpdesk",
        "url": "https://its.ksu.edu.sa/",
        "children": audience_nodes
    }

    return tree

async def scrape_college_details(session, path):
    """
    Given a college URL, return its six modules:
      – About College (with your existing fallbacks)
      – Academic Departments (now with children: title, url, content, faculty_links, contact_info)
      – News, Events…, Service, Important links (unchanged)
    """
    college_base_url = path if path.startswith("http") else urljoin(BASE, path)
    soup = await get_soup(session, path)
    modules = []

    # — About College —
    about = ""
    block = soup.select_one("div.field--name-body")
    if block:
        ps = block.find_all("p")
        about = " ".join(p.get_text(" ", strip=True) for p in ps if p.get_text(strip=True))

    if not about:
        span_block = soup.select_one("span.views-field.views-field-body span.field-content")
        if span_block:
            about = span_block.get_text(" ", strip=True)

    if not about:
        hdr = soup.find(lambda t: t.name in ["h2", "h3", "h4"] and "About College" in t.text)
        if hdr:
            p = hdr.find_next_sibling("p")
            if p:
                about = p.get_text(" ", strip=True)

    if not about:
        for p in soup.find_all("p"):
            t = p.get_text(" ", strip=True)
            if t.startswith("The College of"):
                about = t
                break

    modules.append({"section": "About College", "content": about})

    # — Academic Departments —
    dept_links = []
    seen_urls = set()
    keywords = ["academic departments", "departments", "department", "academic"]
    hdr = soup.find(lambda t: t.name in ["h2", "h3", "h4"] and any(
        kw in t.get_text(strip=True).lower() for kw in keywords)
    )

    if "sciences.ksu.edu.sa" in path:
        print("🔁 Special handling for College of Sciences")

        soup = await get_soup(session, path)
        base = path if path.endswith("/") else path + "/"
        departments = []

        # Step 1: From main page, find departments from dropdown menu
        for a in soup.select("li.menu-item a[href]"):
            text = a.get_text(strip=True).lower()
            if "department" in text and "/en/" in a["href"]:
                dept_url = urljoin(base, a["href"])
                dept_title = a.get_text(strip=True)
                print(f"📁 Found department: {dept_title}")

                try:
                    dept_soup = await get_soup(session, dept_url)
                    about = await fetch_about_of(session, dept_url)
                    contact_link = find_contact_link(dept_soup, dept_url)
                    contact_info = await extract_contact_info(session, contact_link) if contact_link else "Contact page not found."

                    # Step 2: Inside department page, find EDUCATION dropdown menu
                    edu_links = []
                    for edu_a in dept_soup.select("li.menu-item a[href]"):
                        edu_text = edu_a.get_text(strip=True).lower()
                        if any(x in edu_text for x in ["faculty", "staff", "employee"]):
                            edu_url = urljoin(dept_url, edu_a["href"])
                            edu_links.append({
                                "title": edu_a.get_text(strip=True),
                                "url": edu_url
                            })
                            print(f"👥 Found faculty/staff link: {edu_url}")

                    departments.append({
                        "title": dept_title,
                        "url": dept_url,
                        "content": about,
                        "contact_info": contact_info,
                        "faculty_staff_links": edu_links
                    })

                except Exception as e:
                    print(f"❌ Error processing {dept_url}: {e}")

        modules = []
        if departments:
            modules.append({
                "section": "Academic Departments",
                "children": departments
            })

        return modules

    grid = hdr.find_next("div", class_="views-view-grid") if hdr else None

    if grid:
        # Process departments concurrently
        async def process_department(a):
            title = a.get_text(strip=True)
            href = a["href"].strip()
            if not title:
                return None
            if href.startswith("/ar/") and "/en/" not in href:
                href = href.replace("/ar/", "/en/")
            href = href if href.startswith("http") else urljoin(college_base_url, href)
            if href in seen_urls:
                return None
            seen_urls.add(href)

            try:
                content_task = fetch_about_of(session, href)
                dep_soup_task = get_soup(session, href)
                
                content, dep_soup = await asyncio.gather(content_task, dep_soup_task)
            except Exception as e:
                print(f"Skipping {href} due to error: {e}")
                return None

            contact_link = find_contact_link(dep_soup, href)
            
            # Run contact extraction and faculty links concurrently
            contact_task = extract_contact_info(session, contact_link) if contact_link else asyncio.create_task(asyncio.sleep(0, result="Contact page not found."))
            faculty_task = find_faculty_links(session, href)
            
            contact_info, faculty_links = await asyncio.gather(contact_task, faculty_task)

            return {
                "title":   title,
                "url":     href,
                "content": content,
                "faculty_links": faculty_links,
                "contact_info": contact_info
            }

        # Process all departments concurrently
        dept_tasks = [process_department(a) for a in grid.select(".portfolio-content a[href]")]
        dept_results = await asyncio.gather(*dept_tasks, return_exceptions=True)
        
        # Filter out None results and exceptions
        dept_links = [result for result in dept_results if result is not None and not isinstance(result, Exception)]
    else:
        print(f"⚠️ Fallback (JS-rendered): using Selenium on {college_base_url}")
        try:
            js_departments = await asyncio.to_thread(extract_js_departments, college_base_url)
            
            async def process_js_department(dep):
                title = dep["title"]
                href = dep["url"]
                if href in seen_urls:
                    return None
                seen_urls.add(href)
                print(f"🧲 JS dept found: {title} → {href}")
                
                try:
                    dep_soup = await get_soup(session, href)
                except Exception as e:
                    print(f"❌ Skipping {href} due to error: {e}")
                    return None
                    
                content = await fetch_about_of(session, href)
                contact_link = find_contact_link(dep_soup, href)
                
                contact_task = extract_contact_info(session, contact_link) if contact_link else asyncio.create_task(asyncio.sleep(0, result="Contact page not found."))
                faculty_task = find_faculty_links(session, href)
                
                contact_info, faculty_links = await asyncio.gather(contact_task, faculty_task)
                
                return {
                    "title":   title,
                    "url":     href,
                    "content": content,
                    "faculty_links": faculty_links,
                    "contact_info": contact_info
                }

            js_dept_tasks = [process_js_department(dep) for dep in js_departments]
            js_dept_results = await asyncio.gather(*js_dept_tasks, return_exceptions=True)
            
            dept_links = [result for result in js_dept_results if result is not None and not isinstance(result, Exception)]
            
        except Exception as e:
            print(f"⚠️ JS Fallback failed: {e}")

    modules.append({
        "section":  "Academic Departments",
        "children": dept_links
    })

    # — Remaining: News, Events…, Service, Important links —
    for name in ["Service"]:
        items = []
        hdr2 = soup.find(lambda t: t.name in ["h2", "h3", "h4"] and name in t.text)
        if hdr2:
            view2 = hdr2.find_next_sibling("div", class_="view-content")
            if view2:
                for a in view2.find_all("a", href=True):
                    items.append({
                        "title": a.get_text(strip=True),
                        "url":   a["href"].strip()
                    })
        modules.append({"section": name, "items": items})

    return modules

async def scrape_and_update():
    session = await create_session()
    
    try:
        print("🚀 Starting KSU website scraping...")
        print(f"📁 Output directory: {OUTPUT_DIR}")
        print(f"📄 Output file: {OUTPUT_FILE}")
        
        # 1) scrape menu
        print("📋 Scraping main menu...")
        menu = await scrape_menu(session)

        # Create tasks for major sections that can run concurrently
        print("📘 Scraping major sections concurrently...")
        regulations_task = scrape_regulations_and_policies(session)
        admission_task = scrape_admission_requirements(session)
        faq_task = merge_all_faqs(session)
        research_task = scrape_research_institutes(session)
        library_task = scrape_library_section(session)
        calendar_task = scrape_academic_calendar(session)
        housing_task = scrape_housing_section(session)
        
        # Wait for all major sections to complete
        (regulations_section, admission_req_section, faq_section, 
         research_institutes, library_section, academic_calendar_section, 
         housing_section) = await asyncio.gather(
            regulations_task, admission_task, faq_task, research_task,
            library_task, calendar_task, housing_task,
            return_exceptions=True
        )

        # Add sections to menu
        if not isinstance(regulations_section, Exception) and regulations_section["children"]:
            menu.append(regulations_section)
            print("✅ Added Regulations and Policies section")
        else:
            print("⚠️ No policies found under Regulations and Policies.")

        if not isinstance(admission_req_section, Exception) and admission_req_section["children"]:
            menu.append(admission_req_section)
            print("✅ Added Admission Requirements section")
        else:
            print("⚠️ No admissions found.")

        if not isinstance(faq_section, Exception) and faq_section:
            menu.extend(faq_section)
            print("✅ Added FAQs section")
        else:
            print("⚠️ No FAQs found.")

        if not isinstance(research_institutes, Exception):
            labs_children = [
                {"title": inst["title"], "url": "https://ksu.edu.sa/en/node/3106", "content": ""}
                for inst in research_institutes
            ]
            # manually append Central Research Lab
            labs_children.append({
                "title": "Central Research Lab",
                "url": "https://crl.ksu.edu.sa/en",
                "content": ""
            })

            research_section = {
                "title": "Research",
                "url": "https://ksu.edu.sa/en/node/3106",
                "content": "Research institutes at King Saud University.",
                "children": [
                    {
                        "title": "Labs",
                        "url": "https://ksu.edu.sa/en/node/3106",
                        "children": labs_children
                    }
                ]
            }
            menu.append(research_section)
            print("✅ Added Research section")

        # Add other sections
        if not isinstance(library_section, Exception) and library_section:
            menu.append(library_section)
            print("✅ Added Libraries section")
        else:
            print("❌ Library section failed.")

        if not isinstance(academic_calendar_section, Exception) and academic_calendar_section:
            menu.append(academic_calendar_section)
            print("✅ Added Academic Calendar section")

        if not isinstance(housing_section, Exception) and housing_section:
            menu.append(housing_section)
            print("✅ Added Housing section")

        # IT Helpdesk (synchronous operation)
        print("💻 Processing IT Helpdesk data...")
        csv_path = "IT_Helpdesk_cleaned.csv"
        try:
            it_helpdesk_section = build_it_helpdesk_tree(csv_path)
            if it_helpdesk_section:
                menu.append(it_helpdesk_section)
                print("✅ Added IT Helpdesk section")
        except FileNotFoundError:
            print(f"⚠️ CSV file '{csv_path}' not found. Skipping IT Helpdesk section.")
        except Exception as e:
            print(f"❌ Error processing IT Helpdesk: {e}")

        # 2) drill into Study at KSU → Colleges
        print("🏫 Processing colleges data...")
        try:
            study = next(m for m in menu if m["title"].lower() == "study at ksu")
            colleges_node = next(c for c in study["children"] if c["title"].lower() == "colleges")

            # 3) build each category → colleges → details
            for cat in colleges_node["children"]:
                print(f"📚 Processing category: {cat['title']}")
                cat["children"] = await scrape_category(session, cat["url"])
                
                # Process all colleges in this category concurrently
                college_tasks = [scrape_college_details(session, coll["url"]) for coll in cat["children"]]
                college_results = await asyncio.gather(*college_tasks, return_exceptions=True)
                
                # Assign results to colleges
                for i, result in enumerate(college_results):
                    if not isinstance(result, Exception):
                        cat["children"][i]["children"] = result
                        print(f"  ✅ Processed college: {cat['children'][i]['title']}")
                    else:
                        print(f"  ❌ Error processing college {cat['children'][i]['title']}: {result}")
                        cat["children"][i]["children"] = []
        except StopIteration:
            print("⚠️ Could not find 'Study at KSU' or 'Colleges' section in menu")

        # 4) Save to JSON file
        print("💾 Saving data to JSON file...")
        save_to_json(menu)
        
        print("🎉 Scraping completed successfully!")
        print(f"📊 Total menu items: {len(menu)}")

    except Exception as e:
        print(f"❌ Critical error in main function: {e}")
        raise
    finally:
        await session.close()
        print("🔒 Session closed")

if __name__ == "__main__":
    asyncio.run(scrape_and_update())



