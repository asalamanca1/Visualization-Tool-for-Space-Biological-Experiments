import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import './Home.css';
import '../App.css';
import HeroSection from './HeroSection';
import Footer from './Footer';



function Home() {
  const [url, setUrl] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const navigate = useNavigate();
  const [loadingText, setLoadingText] = useState('Loading');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const response = await axios.post('http://localhost:5001/api/submit_url', { url });
      if (response.data.experiment_id) {
        // Navigate to the experiment details page
        navigate(`/experiment/${response.data.experiment_id}`);
      } else if (response.data.error) {
        setError(response.data.error);
      }
    } catch (err) {
      setError(err.response?.data?.error || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    let interval;
    if (loading) {
      let dots = '';
      interval = setInterval(() => {
        dots = dots.length < 3 ? dots + '.' : '';
        setLoadingText(`Loading${dots}`);
      }, 500);
    } else {
      setLoadingText('Loading');
    }
    return () => clearInterval(interval);
  }, [loading]);


  return (
    <>
      {loading ? (
        // Render the animated loading screen
        <div className="loading-screen">
          <div className="loading-content">
            {/* Animation container */}
            <div className="container">
              <div className="moon">
                <div className="crater crater-1"></div>
                <div className="crater crater-2"></div>
                <div className="crater crater-3"></div>
                <div className="crater crater-4"></div>
                <div className="crater crater-5"></div>
                <div className="shadow"></div>
                <div className="eye eye-l"></div>
                <div className="eye eye-r"></div>
                <div className="mouth"></div>
                <div className="blush blush-1"></div>
                <div className="blush blush-2"></div>
              </div>
              <div className="orbit">
                <div className="rocket">
                  <div className="window"></div>
                </div>
              </div>
            </div>
            {/* Loading text */}
            <div className="loading-text">{loadingText}</div>
          </div>
        </div>
      ) : (
        // Render the normal content when not loading
        <>
          <HeroSection>
            
            <div className="container">
            <h1>Interstellar Automated</h1>
            <h1>Visualizer</h1>
              <form onSubmit={handleSubmit} className="url-form">
                <input
                  type="text"
                  value={url}
                  onChange={(e) => setUrl(e.target.value)}
                  placeholder="Enter the URL of the experiment"
                  required
                />
                <button type="submit" disabled={loading}>
                  {loading ? 'Processing...' : 'Submit'}
                </button>
              </form>
              {error && <p className="error-message">{error}</p>}
            </div>
          </HeroSection>
          <div id="wrap">
        <Footer />
      </div>
        </>
      )}
    </>
  );
}

export default Home;
