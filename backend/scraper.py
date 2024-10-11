from playwright.sync_api import sync_playwright, TimeoutError
import requests
import os

def scrape_data(url):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")
        
        # Scrape header (summary text)
        header = page.query_selector('p.ws-pre-line')
        header_text = header.inner_text().strip() if header else "Header not found"
        
        span_element = page.query_selector("h1 > span.ng-star-inserted:not([class*=' '])")
        sub_header = span_element.inner_text().strip() if span_element else "Span text not found"
        
        # Scrape project information
        project_info = scrape_project_info(page)
        
        # Scrape table data
        table_data = scrape_table_data(page)

        contacts = scrape_contacts(page)
        
        # Scrape organism information
        organism_element = page.query_selector('mat-grid-tile a')
        organism = organism_element.inner_text().strip() if organism_element else ""
        
            # Scrape publications
        publications_element = page.query_selector('publications-panel')
        publications_text = publications_element.inner_text().strip() if publications_element else "Publications not found"
        
        
        # Scrape mission dates
        mission_dates = scrape_mission_dates(page)
        
        
        browser.close()

    return {
        "header": header_text,
        "project_info": project_info,
        "table_data": table_data,
        "organism": organism,
        "mission_dates": mission_dates,
        "sub_header": sub_header,
        "contacts": contacts,
        "publications_test": publications_text 
    }


def scrape_project_info(page):
   # Scrape Project Information
   project_info = {}
   keys = [
       ("Payload Identifier", "Payload Identifier"),
       ("Project Title", "Project Title"),
       ("Project Type", "Project Type"),
       ("Flight Program", "Flight Program"),
       ("Experiment Platform", "Experiment Platform"),
       ("Sponsoring Agency", "Sponsoring Agency"),
       ("NASA Center", "NASA Center"),
       ("Funding Source", "Funding Source"),
       ("Additional Project Metadata", "Additional Project Metadata", True)
   ]
  
   for key_tuple in keys:
       label, key = key_tuple[:2]
       link = key_tuple[2] if len(key_tuple) > 2 else False
       selector = f'td.mat-cell:has-text("{label}") + td.mat-cell'
       if link:
           element = page.query_selector(f'{selector} a')
           if element:
            project_info[key] = element.inner_text().strip() if element.inner_text() else ""
            project_info[f'{key} Link'] = element.get_attribute('href') if element.get_attribute('href') else ""
       else:
           element = page.query_selector(selector)
           project_info[key] = element.inner_text().strip() if element and element.inner_text() else ""
  
   return project_info if project_info else {}

def scrape_contacts(page):
    """Scrape the contact name and email from the page for the first contact only"""
    contacts = []

    # Find all contact containers
    contact_blocks = page.query_selector_all('span.contact-container')

    if contact_blocks:
        # Process only the first contact block
        first_contact_block = contact_blocks[0]

        # Extract the name from the <a> tag
        name_element = first_contact_block.query_selector('a')
        name = name_element.inner_text().strip() if name_element else "Name not found"

        # Click on the name to reveal the email (if necessary)
        if name_element:
            name_element.click()
            # Wait for the email to be revealed after the click
            page.wait_for_timeout(1000)  # Small delay to ensure the email is revealed

        # Extract the email from the <a> tag with href="mailto"
        email_span = first_contact_block.query_selector('span:has(a[href^="mailto:"])')
        email_element = email_span.query_selector('a[href^="mailto:"]') if email_span else None
        email = email_element.get_attribute('href').replace("mailto:", "").strip() if email_element else "Email not found"

        # Append the first contact's details to the list
        contacts.append({
            "name": name,
            "email": email
        })

    return contacts if contacts else []

def scrape_table_data(page):
    # Scrape the table rows within <tbody>
    rows = page.query_selector_all('tbody[role="rowgroup"] tr')
    table_data = []

    for row in rows:
        # Extract data for the "Factor" and "Concept" table rows
        factor_element = row.query_selector('td.mat-column-factor')
        concept_element = row.query_selector('td.mat-column-concept a') or row.query_selector('td.mat-column-concept')

        # Extract data for the "Measurement", "Technology", and "Platform" table rows
        measurement_element = row.query_selector('td.mat-column-measurement a')
        technology_element = row.query_selector('td.mat-column-technology a')
        platform_element = row.query_selector('td.mat-column-platform')

        # Handling "Factor" and "Concept"
        if factor_element and concept_element:
            factor = factor_element.inner_text().strip()
            concept = concept_element.inner_text().strip()
            table_data.append((factor, concept))

        # Handling "Measurement", "Technology", and "Platform"
        elif measurement_element and technology_element and platform_element:
            measurement = measurement_element.inner_text().strip()
            technology = technology_element.inner_text().strip()
            platform = platform_element.inner_text().strip()
            table_data.append((measurement, technology, platform))

    return table_data

def scrape_mission_dates(page):
    # Scrape the mission date rows within <tbody>
    rows = page.query_selector_all('tbody[role="rowgroup"] tr')
    mission_dates = []

    for row in rows:
        identifier_element = row.query_selector('td.mat-column-identifier a')
        start_date_element = row.query_selector('td.mat-column-startDate')
        end_date_element = row.query_selector('td.mat-column-endDate')

        # Extract the data only if all elements are present
        if identifier_element and start_date_element and end_date_element:
            identifier = identifier_element.inner_text().strip()
            start_date = start_date_element.inner_text().strip()
            end_date = end_date_element.inner_text().strip()
            mission_dates.append({
                "Identifier": identifier,
                "Start Date": start_date,
                "End Date": end_date
            })
    
    return mission_dates

def download_and_append_csv(url, desired_path, assays_path):
    desired_dir = os.path.dirname(desired_path)
    assays_dir = os.path.dirname(assays_path)

    if not os.path.exists(desired_dir):
        os.makedirs(desired_dir)
        print(f"Created directory: {desired_dir}")

    if not os.path.exists(assays_dir):
        os.makedirs(assays_dir)
        print(f"Created directory: {assays_dir}")
    if os.path.exists(desired_path) and os.path.exists(assays_path):
        # Wipe the files before starting the download process
        wipe_file(desired_path)
        wipe_file(assays_path)

    with sync_playwright() as p:
        experiment_title = url.split('/')[-1]
        browser = p.chromium.launch(headless=True)  # Run headless (without opening the browser)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_load_state("networkidle")

        # First loop for downloading main CSV (samples section)
        process_page_data(page, experiment_title, desired_path, first_time=True, for_assays=False)

        # Second loop for downloading "assays_scraped.csv" (assays section)
        process_page_data(page, experiment_title, assays_path, first_time=True, for_assays=True)

        browser.close()

def process_page_data(page, experiment_title, csv_path, first_time=True, for_assays=False):
    page_number = 1

    while True:
        try:
            print(f"Processing page {page_number} for {'assays_scraped.csv' if for_assays else 'samples CSV'}")

            # Use specific selectors based on whether we are in the samples or assays section
            if first_time:
                if not for_assays:
                    # For samples, target the first button that matches the text
                    export_button = page.locator('button:has-text("Select Export Columns")').first
                else:
                    # Keep the existing selector for assays
                    export_button = page.locator('#cdk-accordion-child-6 button:has-text("Select Export Columns")')

                if export_button.is_visible():
                    export_button.click()
                    page.wait_for_timeout(2000)  # Wait for columns to load
                    print(f"Clicked 'Select Export Columns' button for {'assays' if for_assays else 'samples'} section.")
                else:
                    print(f"Could not find 'Select Export Columns' button for {'assays' if for_assays else 'samples'} section.")
                    break

                first_time = False  # Set to False after the first click

            # Click "Export CSV" button based on the same container but different button text
            if not for_assays:
                export_csv_button = page.locator('button:has-text("Export CSV")').first  # For samples, target the first match
            else:
                # Keep the existing selector for assays
                export_csv_button = page.locator('#cdk-accordion-child-6 button:has-text("Export CSV")')

            if export_csv_button.is_visible():
                with page.expect_download() as download_info:
                    export_csv_button.click()
                print(f"Clicked 'Export CSV' button for {'assays' if for_assays else 'samples'} section.")

                # Save the download to a temporary file first
                temp_csv_path = f"temp_{experiment_title}_{'assays' if for_assays else 'samples'}_page{page_number}.csv"
                download = download_info.value
                download.save_as(temp_csv_path)
                print(f"Temporary CSV downloaded to: {temp_csv_path}")

                # Append the content of the temporary CSV to the correct CSV file
                append_to_csv(temp_csv_path, csv_path)

                # Remove the temporary CSV after appending
                os.remove(temp_csv_path)

            else:
                print(f"Could not find 'Export CSV' button for {'assays' if for_assays else 'samples'} section.")
                break

            # Explicitly wait for the next page button in assays section
            if for_assays:
                print("Looking for 'Next page' button for assays...")

                # Keep the existing selector for the next button in assays
                next_button = page.locator('#cdk-accordion-child-6 button.mat-paginator-navigation-next.mat-icon-button')

                # Increase the timeout and explicitly wait for the button
                next_button.wait_for(state='visible', timeout=10000)  # Wait up to 10 seconds

                if next_button.is_enabled():
                    print("'Next page' button found and enabled. Clicking...")
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                    page_number += 1
                    print(f"Moving to page {page_number} for assays")
                else:
                    print("'Next page' button is disabled or not visible.")
                    break

            else:
                next_button = page.locator('button.mat-paginator-navigation-next.mat-icon-button').first

                # Ensure the next button exists and is visible
                if next_button.is_visible() and next_button.is_enabled():
                    next_button.click()
                    page.wait_for_load_state("networkidle")
                    page_number += 1
                    print(f"Moving to page {page_number} for samples")
                else:
                    print("No more pages or 'Next page' button is not visible.")
                    break

        except TimeoutError:
            print("Timed out while waiting for a page element.")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            break

def append_to_csv(temp_csv_path, desired_path):
    """Appends the content of temp_csv_path to desired_path"""
    with open(temp_csv_path, 'r') as temp_file:
        # Read the header from the temporary file
        header = temp_file.readline()
        data = temp_file.read()

    # Check if the main CSV file already exists
    file_exists = os.path.isfile(desired_path)

    # Open the desired CSV in append mode
    with open(desired_path, 'a') as main_file:
        # Write the header only if the file does not already exist
        if not file_exists:
            main_file.write(header)
        # Append the data
        main_file.write(data)

    print(f"Data from {temp_csv_path} appended to {desired_path}")

def wipe_file(file_path):
    """Wipes the content of a file by opening it in write mode and closing it immediately."""
    with open(file_path, 'w') as file:
        pass  # Opening a file in 'w' mode clears its content

def scrape_contacts(page):
    """Scrape the contact name and email from the page for the first contact only"""
    contacts = []
    
    try:
        # Wait explicitly for the contact container to be available
        page.wait_for_selector('span.contact-container', timeout=5000)  # Wait up to 5 seconds

        # Find all contact containers
        contact_blocks = page.query_selector_all('span.contact-container')

        if len(contact_blocks) == 0:
            # If no contact blocks were found, log this and return an empty list
            print("No contact blocks found on the page.")
            return contacts

        # Process only the first contact block
        first_contact_block = contact_blocks[0]

        # Extract the name from the <a> tag
        name_element = first_contact_block.query_selector('a')
        name = name_element.inner_text().strip() if name_element else "Name not found"

        # Click on the name to reveal the email (if necessary)
        if name_element:
            name_element.click()
            # Wait for the email to be revealed after the click
            page.wait_for_timeout(1000)  # Small delay to ensure the email is revealed

        # Extract the email from the <a> tag with href="mailto"
        email_span = first_contact_block.query_selector('span:has(a[href^="mailto:"])')
        email_element = email_span.query_selector('a[href^="mailto:"]') if email_span else None
        email = email_element.get_attribute('href').replace("mailto:", "").strip() if email_element else "Email not found"

        # If both name and email are not found, log this information
        if name == "Name not found" or email == "Email not found":
            print(f"Primary contact not found: Name = {name}, Email = {email}")
        else:
            # Log the extracted contact for debugging
            print(f"Primary contact extracted: Name = {name}, Email = {email}")

            # Append the first contact's details to the list
            contacts.append({
                "name": name,
                "email": email
            })

    except TimeoutError:
        print("Timeout while waiting for the contact container to appear.")
    
    return contacts


# Example usage
if __name__ == '__main__':
    url = 'https://osdr.nasa.gov/bio/repo/data/studies/OSD-379'
    desired_csv_path = '/Users/aquas/Desktop/hackathon/science-visualizer/backend/scraped.csv'

   

    # Scrape the header, project info, table data, and organism info
    scraped_data = scrape_data(url)

    # Print the scraped data
    print("\nHeader:\n", scraped_data['header'])
    print("\nProject Info:\n", scraped_data['project_info'])
    print("\nTable Data:\n", scraped_data['table_data'])
    print("\nOrganism:\n", scraped_data['organism'])
    print("\nContacts:\n", scraped_data['contacts'])

    # Print the mission dates
    print("\nMission Dates:")
    for mission in scraped_data['mission_dates']:
        print(f"Identifier: {mission['Identifier']}, Start Date: {mission['Start Date']}, End Date: {mission['End Date']}")
