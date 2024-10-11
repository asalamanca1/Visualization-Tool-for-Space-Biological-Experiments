import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { useParams } from 'react-router-dom';
import './ExperimentDetails.css';
import Footer from './Footer';
import Popup from './Popup'; // Import the Popup component

// Chatbot \\
import Chatbot from 'react-chatbot-kit';
import 'react-chatbot-kit/build/main.css';
import config from './chatbot/config';
import MessageParser from './chatbot/MessageParser';
import ActionProvider from './chatbot/ActionProvider';

import chatbotGif from '../images/chatbot.gif';


function ExperimentDetails() {
  const [isChatOpen, setIsChatOpen] = useState(false); // State to toggle chatbot visibility
  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const { id } = useParams();
  const [experiment, setExperiment] = useState(null);
  const [error, setError] = useState(null);
  const [isPopupVisible, setIsPopupVisible] = useState(false); // State for Overview pop-up visibility
  const [isSamplePopupVisible, setIsSamplePopupVisible] = useState(false); // State for Sample Characteristics pop-up visibility
  const [piContact, setPiContact] = useState(null); // State for holding PI contact details
  const [isPiPopupVisible, setIsPiPopupVisible] = useState(false); // State for PI pop-up visibility
  const [sampleCharacteristics, setSampleCharacteristics] = useState(null); // State to hold the sample characteristics data
  const [extractedValue, setExtractedValue] = useState(''); // State to store the extracted value from the URL
  const [subHeader, setSubHeader] = useState('');
  const [showContactsModal, setShowContactsModal] = useState(false); // Primary contact modal
  const [primaryContact, setPrimaryContact] = useState(null); // Primary contact details
  const [rawContacts, setRawContacts] = useState(null); // Raw contacts data
  const [isFactorsPopupVisible, setIsFactorsPopupVisible] = useState(false);
  const [factorData, setFactorData] = useState(null);
  const [isPublicationPopupVisible, setIsPublicationPopupVisible] = useState(false);
  const [publicationsList, setPublicationText] = useState([]);

  useEffect(() => {
    const fetchExperiment = async () => {
      try {
        const response = await axios.get(`http://localhost:5001/api/experiments/${id}`);
        setExperiment(response.data);
        
        const contacts = response.data.contacts || [];
        setRawContacts(contacts);


        if (contacts.length > 0) {
          setPrimaryContact(contacts[0]); // Assuming the first contact is primary
        }

        const extracted = response.data.url.substring(response.data.url.lastIndexOf('/') + 1);
        setExtractedValue(extracted); // Store the extracted value in the state


        const subHeaderValue = response.data.text_content.sub_header;
        setSubHeader(subHeaderValue); // Store the sub_header in the state

        const publicationTextValue = response.data.text_content.publications_test; // Extract publication_text
        const publicationsListValue = extractPublications(publicationTextValue);
        setPublicationText(publicationsListValue); // Store the publication_text in the state
        
      } catch (err) {
        setError(err.response?.data?.error || 'An error occurred.');
      }
    };

    const fetchFactorData = async () => {
        try {
          const response = await axios.get(`http://localhost:5001/api/experiments/${id}/factor_values`);
          console.log("Factor data:", response.data);  // Debugging: Check if the factor data is available
          setFactorData(response.data.data);
        } catch (err) {
          console.error('Error fetching factor data:', err);
          setError('Failed to load factor data.');
        }
      };

    fetchExperiment();
    fetchFactorData();
  }, [id]);

  // Handle Overview pop-up
  const handleOverviewClick = () => {
    setIsPopupVisible(true); // Show Overview pop-up
  };

 
  function extractPublications(publicationsText) {
    const publications = [];
    // Adjusted regex to handle new lines more flexibly
    const regex = /^(.*?)\nAuthors:\s*(.*?)\nDOI:\s*(.*?)\n/gm;
    let match;

    // Loop to find all matches in the input text
    while ((match = regex.exec(publicationsText)) !== null) {
        const publication = {
            title: match[1].trim(),
            authors: match[2].trim(),
            doi: match[3].trim()
        };
        publications.push(publication);
    }

    return publications;
}


  const handleButtonClick = async () => {
    console.log(publicationsList);
    setIsPublicationPopupVisible(true); // Show the popup
  };  

  // Handle Sample Characteristics pop-up and fetch the sample characteristics data from the API
  const handleSamplePopupClick = async () => {
    try {
      const response = await axios.get(`http://localhost:5001/api/experiments/${id}/organism_characteristics`);
      setSampleCharacteristics(response.data); // Set the fetched data
      setIsSamplePopupVisible(true); // Show the pop-up after fetching the data
    } catch (err) {
      setError('Error fetching sample characteristics.');
    }
  };

  // Handle PI pop-up and fetch PI contact details from the API
  const handlePiPopupClick = async () => {
    try {
      const contactInfo = textContent.contacts && textContent.contacts.length > 0 ? textContent.contacts[0] : null;
      setPiContact(contactInfo); // Set the PI contact details
      setIsPiPopupVisible(true); // Show the pop-up with the contact info
    } catch (err) {
      setError('Error fetching PI contact information.');
    }
  };

  // Close Pop-ups
  const closePopup = () => {
    setIsPopupVisible(false); // Hide Overview pop-up
  };

  const closeSamplePopup = () => {
    setIsSamplePopupVisible(false); // Hide Sample Characteristics pop-up
  };

  const closePiPopup = () => {
    setIsPiPopupVisible(false); // Hide PI pop-up
  };

  const handleFactorsClick = () => {
    setIsFactorsPopupVisible(true);
  };
  
  const closeFactorsPopup = () => {
    setIsFactorsPopupVisible(false);
  };

  if (error) {
    return <p className="error-message">{error}</p>;
  }

  if (!experiment) {
    return <p>Loading experiment details...</p>;
  }

  const textContent = experiment.text_content || {};
  const projectInfo = textContent.project_info || {};
  const projectTitle = projectInfo['Project Title'] || NaN;

  const generateWikiLink = (term) => {
    if (term && term !== 'N/A') {
      const sanitizedTerm = term.replace(/\s*\(.*?\)\s*/g, '').trim();
      return `https://en.wikipedia.org/wiki/${encodeURIComponent(sanitizedTerm)}`;
    }
    return '';
  };

  const overviewContent = (
    <>
      <p>{textContent.header || 'No header available.'}</p>
    </>
  );

    // Sample Characteristics pop-up content
    const sampleContent = sampleCharacteristics && sampleCharacteristics.columns && sampleCharacteristics.data ? (
    <>
        <ul>
        {sampleCharacteristics.data.length > 0 ? (
            sampleCharacteristics.data.map((row, idx) => (
            <li key={idx}>
                {sampleCharacteristics.columns.map((col, colIdx) => {
                // Strip "characteristics" from the key, trim spaces, and remove any leading colon or whitespace
                const strippedKey = col.replace(/characteristics/i, '').trim().replace(/^[:\s]+/, '');
                const value = row[col];

                return (
                    <div key={colIdx}>
                    {strippedKey ? (
                        <>
                        <strong>{strippedKey}:</strong> {value}
                        </>
                    ) : (
                        <>{value}</>
                    )}
                    </div>
                );
                })}
            </li>
            ))
        ) : (
            <p>No sample characteristics available.</p>
        )}
        </ul>
    </>
    ) : (
    <p>Loading sample characteristics...</p>
    );

  // PI Contact Information pop-up content
  const piContent = piContact ? (
    <>
      <h3>PI Contact Information</h3>
      <p><strong>Name:</strong> {piContact.name}</p>
      <p><strong>Email:</strong> <a href={`mailto:${piContact.email}`}>{piContact.email}</a></p>
    </>
  ) : (
    <p>No PI contact information available.</p>
  );

  const botConfig = {
    ...config,
    initialState: {
      experimentId: id,
    },
  };

  const factorsContent = (
    <>
      <ul>
        {factorData && Object.keys(factorData).length > 0 ? (
          Object.entries(factorData).map(([key, values], index) => {
            // Strip "factor value" from the key, trim spaces, and remove any leading colon or whitespace
            const strippedKey = key.replace(/factor value/i, '').trim().replace(/^[:\s]+/, '');
  
            return (
              <li key={index}>
                {strippedKey ? (
                  <>
                    <strong>{strippedKey}:</strong> {values.join(', ')}
                  </>
                ) : (
                  <>{values.join(', ')}</>
                )}
              </li>
            );
          })
        ) : (
          <p>No experimental factors available.</p>
        )}
      </ul>
    </>
  );
  


  console.log("Experiment ID:", id); 

  let header = projectTitle ? `${extractedValue}: ${projectTitle}` : `${extractedValue}`;
  
  return (
    <div className="experiment-container">
      {/* Title card */}
      <div className="card-title">
        <h1>{header}</h1>
        <h2>{subHeader}</h2>
      </div>

      {/* Sidebar and dashboard aligned horizontally */}
      <div className="bottom-section">
        <div className="sidebar">
          <h2>Experimental Overview</h2>
          <ul>
            <li>
              <button className="sidebar-button" onClick={handleOverviewClick}>
                Description
              </button>
            </li>
            <li>
              <button className="sidebar-button" onClick={handleSamplePopupClick}>
                Subject Characteristics
              </button>
            </li>
            <li>
                <button className="sidebar-button" onClick={handleFactorsClick}>
                    Experimental Factors
                </button>
            </li>
            <li>
              <button className="sidebar-button" onClick={() => window.open(generateWikiLink(projectInfo['NASA Center']), '_blank')}>
                NASA Center: {projectInfo['NASA Center'] || 'N/A'}
              </button>
            </li>
            <li>
              <button className="sidebar-button" onClick={() => window.open(generateWikiLink(projectInfo['Flight Program']), '_blank')}>
                Flight Program: {projectInfo['Flight Program'] || 'N/A'}
              </button>
            </li>
            <li>
              <button className="sidebar-button" onClick={handlePiPopupClick}>
                Show PI Contact Information
              </button>
            </li>
            <li>
              <button className="sidebar-button" onClick={handleButtonClick}>
                Publication(s) 
              </button>
            </li>
          </ul>
        </div>

        {/* Dashboard content */}
        <div className="dashboard-content" style={{ backgroundColor: 'black', padding: '8px', borderRadius: '16px', maxWidth: '820px', maxHeight: '800px', overflow: 'hidden' }}>
          <iframe
            src={`http://localhost:5001/dash/${id}`}
            style={{
              width: '100%',
              height: '100%',
              border: '2px solid #ffffff',
              borderRadius: '8px',
              transformOrigin: '0 0',
            }}
            title="Dash Dashboard"
          />
        </div>
      </div>

      {/* Pop-up Window for Overview */}
      <Popup isVisible={isPopupVisible} onClose={closePopup} title="Overview">
        {overviewContent}
      </Popup>
      <Popup isVisible={isFactorsPopupVisible} onClose={closeFactorsPopup} title="Experimental Factors">
        {factorsContent}
        </Popup>

      <Popup 
    isVisible={isPublicationPopupVisible} 
    onClose={() => setIsPublicationPopupVisible(false)} 
    title="Publications"
>
    {publicationsList.length > 0 ? (
        publicationsList.map((pub, index) => (
            <div key={index} className="publication-item">
                <h3>{pub.title}</h3>
                <p>Authors: {pub.authors}</p>
                <p>DOI: {pub.doi}</p>
            </div>
        ))
    ) : (
        <p>No publications found.</p>
    )}
</Popup>
      { /* Logging the experiment ID before passing to Chatbot */ }
   

      {/* Pop-up Window for Sample Characteristics */}
      <Popup isVisible={isSamplePopupVisible} onClose={closeSamplePopup} title="Subject Characteristics">
        {sampleContent}
      </Popup>

      {/* Pop-up Window for PI Contact */}
      <Popup isVisible={isPiPopupVisible} onClose={closePiPopup} title="PI Contact Information">
        {piContent}
      </Popup>

      

      {/* Chatbot Toggle Button */}
      <button className="chatbot-toggle-button" onClick={toggleChat}>
        <img src={chatbotGif} alt="Open Chatbot" />
      </button>

      {/* Chatbot Container */}
      {isChatOpen && (
        <div className="chatbot-container">
          <Chatbot
            config={botConfig}
            messageParser={MessageParser}
            actionProvider={ActionProvider}
          />
        </div>
      )}

      <div id="wrap">
        <Footer />
      </div>
    </div>
  );
}

export default ExperimentDetails;
