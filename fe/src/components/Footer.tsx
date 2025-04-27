import React from 'react';
import { Github } from 'lucide-react';

const Footer: React.FC = () => {
  return (
    <footer className="w-full py-6 bg-gray-900 border-t border-gray-800">
      <div className="container mx-auto px-4 flex items-center justify-center space-x-4 text-sm text-gray-400">
        <span>© {new Date().getFullYear()} Telugu Text Converter</span>
        <span className="text-gray-700">•</span>
        <a 
          href="#"
          className="text-indigo-400 hover:text-indigo-300 transition-colors flex items-center"
        >
          <Github className="w-4 h-4 mr-1" />
          Source Code
        </a>
      </div>
    </footer>
  );
};

export default Footer;