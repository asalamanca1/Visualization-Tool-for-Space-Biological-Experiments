import React, { useEffect, useRef } from 'react';
import './Contacts.css';

function Contacts() {
    const trackRef = useRef(null);
    let mouseDownAt = 0;
    let prevPercentage = 0;

    useEffect(() => {
        const track = trackRef.current;
        if (!track) return; // Ensure track exists

        const handleMouseDown = (e) => {
            mouseDownAt = e.clientX;
            track.dataset.mouseDownAt = mouseDownAt;
            prevPercentage = parseFloat(track.dataset.percentage) || 0;
        };

        const handleMouseUp = () => {
            track.dataset.mouseDownAt = "0";
            track.dataset.prevPercentage = track.dataset.percentage;
        };

        const handleMouseMove = (e) => {
            if (track.dataset.mouseDownAt === "0") return;

            const mouseDelta = parseFloat(track.dataset.mouseDownAt) - e.clientX;
            const maxDelta = window.innerWidth / 2;

            const percentage = (mouseDelta / maxDelta) * -100;
            const nextPercentage = Math.min(Math.max(prevPercentage + percentage, -100), 0);

            track.dataset.percentage = nextPercentage;
            track.style.transform = `translate(${nextPercentage}%, -50%)`;

            const images = track.getElementsByClassName("image");
            for (const image of images) {
                image.style.objectPosition = `${100 + nextPercentage}% 50%`;
            }
        };

        // Attach event listeners to the track element
        window.addEventListener('mousedown', handleMouseDown);
        window.addEventListener('mouseup', handleMouseUp);
        window.addEventListener('mousemove', handleMouseMove);

        // Cleanup event listeners on unmount
        return () => {
            window.removeEventListener('mousedown', handleMouseDown);
            window.removeEventListener('mouseup', handleMouseUp);
            window.removeEventListener('mousemove', handleMouseMove);
        };
    }, []);

    return (
        <div className="contacts-page-body">
            <div id="image-track" ref={trackRef} data-mouse-down-at="0" data-percentage="0">
                <a href="https://www.linkedin.com/in/salamanca056/" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D5603AQGgCXZlJPHqLg/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1725161266861?e=1733961600&v=beta&t=WUpr3lDcu1dAy74teLCp5DKAQfQRttRRFnzqvtKiMGI" draggable="false" />
                </a>
                <a href="https://www.linkedin.com/in/kaan-koc-09979a214//" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D5603AQF0yhswjNaz2g/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1718478170057?e=1733961600&v=beta&t=cyQ0W0AnhWvao_j5ZD21500zha-AiX8Nq8wSBxPtHck" draggable="false" />
                </a>
                <a href="https://www.linkedin.com/in/johnpaul-lopez/overlay/photo/" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D4E03AQG7zlewpc0OOQ/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1710783015463?e=1733961600&v=beta&t=GXa5_PIJczOMCRedI4yTPu20BGK_w2n53VDfiwHCcgk" draggable="false" />
                </a>
                <a href="https://www.linkedin.com/in/harlan-t-phillips/" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D5603AQFErSgKDlKi2g/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1705249798564?e=1733961600&v=beta&t=mxD2j57bqQDErZSlOjNROsDfjwnzN9tRsWRoLFTCknM" draggable="false" />
                </a>
                <a href="https://www.linkedin.com/in/carolklingler/" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D4E03AQHO7qBgFXkWnA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1708529434873?e=1733961600&v=beta&t=FPa8JX8iJFZEPI4HzWa9RMe1zRrJk2hUXaihCYeusj8" draggable="false" />
                </a>
                <a href="https://www.linkedin.com/in/matthew-patterson-881b02241/" target="_blank" rel="noopener noreferrer">
                    <img className="image" src="https://media.licdn.com/dms/image/v2/D5603AQE4esdFoUPbgA/profile-displayphoto-shrink_800_800/profile-displayphoto-shrink_800_800/0/1728263060631?e=1733961600&v=beta&t=OJQSJh4XsWj9b4C60RaGDAsbsjcML0mFiVRZyXnZmN8" draggable="false" />
                </a>

                {/* Add more images as needed */}
            </div>
        </div>
    );
}

export default Contacts;
