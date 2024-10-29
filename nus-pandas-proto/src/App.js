import './App.css';
import AppRouter from './routes/routes';
import { AuthProvider } from './context/AuthContext';
import { Amplify } from 'aws-amplify';
import amplifyconfig from './amplifyconfiguration.json';
import SignInModal from './components/modals/SignInModal';
import SignUpModal from './components/modals/SignUpModal';
Amplify.configure(amplifyconfig);


function App() {
  return (
    <AuthProvider>
      <AppRouter />
      <SignInModal />
      <SignUpModal />
    </AuthProvider>
  );
}

export default App;
