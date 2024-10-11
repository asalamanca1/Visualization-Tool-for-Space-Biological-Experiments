import React from 'react';
import './HeroSection.css';
import '../App.css';

function HeroSection({ children }) {
    return (
      <div className="hero-container">
        <video src='/videos/homepage.mp4' autoPlay loop muted />
        <div className="content">
          {children}
        </div>
      </div>
    );
  }
  
  export default HeroSection;
