import React from 'react';
import Header from './components/Header';
import FileUploader from './components/FileUploader';
import TextDisplay from './components/TextDisplay';
import Footer from './components/Footer';
import useTeluguConverter from './hooks/useTeluguConverter';

function App() {
  const { convertedText, isLoading, error, convertImage } = useTeluguConverter();

  const handleFileSelect = (file: File) => {
    convertImage(file);
  };

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      <Header />
      
      <main className="flex-grow container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-white mb-3">
              Telugu Handwriting to Text
            </h2>
            <p className="text-gray-400">
              Convert your Telugu handwritten text to digital format instantly
            </p>
          </div>
          
          {error && (
            <div className="mb-8 p-4 bg-red-900/30 border border-red-700 rounded-lg text-red-300 text-center">
              {error}
            </div>
          )}
          
          <FileUploader onFileSelect={handleFileSelect} />
          <TextDisplay text={convertedText} isLoading={isLoading} />
        </div>
      </main>
      
      <Footer />
    </div>
  );
}

export default App;