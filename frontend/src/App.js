import './App.css';
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Home from './components/Home';
import ExperimentDetails from './components/ExperimentDetails';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import About from './components/About';
import Contacts from './components/Contacts';


function App() {
  return (
    <>
    <Router>
      <Navbar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/experiment/:id" element={<ExperimentDetails />} />
        <Route path='/' exact component={Home}/>
        <Route exact path="/about" element={<About />} />
        <Route exact path="/contacts" element={<Contacts />} />
        
      </Routes>
      </Router>
    </>
  );
}

export default App;
