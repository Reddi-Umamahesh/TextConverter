import React from 'react';
import { FileText } from 'lucide-react';

const Header: React.FC = () => {
  return (
    <header className="w-full py-6 px-4 bg-gray-900 border-b border-gray-800">
      <div className="container mx-auto flex items-center justify-center">
        <FileText className="w-8 h-8 text-indigo-500 mr-3" />
        <h1 className="text-2xl font-bold text-white">Telugu Text Converter</h1>
      </div>
    </header>
  );
};

export default Header;