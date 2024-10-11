// assets/export_cytoscape.js

document.addEventListener('DOMContentLoaded', function() {
    const exportButton = document.getElementById('export-png-button');
    exportButton.addEventListener('click', function() {
        console.log('Export button clicked.');
        // Find the Cytoscape container
        const cyContainer = document.getElementById('cytoscape-tree');
        if (cyContainer && cyContainer._cy) { // '_cy' is the Cytoscape instance
            console.log('Cytoscape instance found.');
            const cy = cyContainer._cy;
            cy.png({
                output: 'blob',
                scale: 2, // Adjust scale for better resolution
                full: true
            }).then(function(blob) {
                console.log('PNG blob created.');
                const url = URL.createObjectURL(blob);
                const link = document.createElement('a');
                link.href = url;
                link.download = 'cytoscape_graph.png';
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
                URL.revokeObjectURL(url);
                console.log('Download triggered.');
            }).catch(function(error) {
                console.error('Error exporting Cytoscape graph:', error);
                alert('Failed to export the graph. Please try again.');
            });
        } else {
            console.error('Cytoscape instance not found.');
            alert('Cytoscape graph is not available for export.');
        }
    });
});
