import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import NavigationBar from '../components/navigation/NavigationBar.jsx';

import Home from '../pages/Home.jsx';
import CGPT from '../pages/CGPT.jsx';
import InsurerDashboard from '../pages/InsurerDashboard.jsx';
import Claim from '../pages/Claim.jsx';
import Policy from '../pages/Policy.jsx';

const AppRouter = () => {
    return (
        <Router>
            <Routes>
                <Route element={<NavigationBar />}>
                    <Route path="/" element={<Home />} />
                    <Route path="/CGPT" element={<CGPT />} />
                    <Route path="/insurer" element={<InsurerDashboard />} />
                    <Route path="/claims" element={<Claim />} />
                    <Route path="/policies" element={<Policy />} />
                </Route>
            </Routes>
        </Router>
    );
}

export default AppRouter;