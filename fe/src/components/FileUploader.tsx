import React, { useState } from 'react';
import { Upload, FileUp, X } from 'lucide-react';

interface FileUploaderProps {
  onFileSelect: (file: File) => void;
}

const FileUploader: React.FC<FileUploaderProps> = ({ onFileSelect }) => {
  const [dragActive, setDragActive] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file: File) => {
    if (file.type.startsWith('image/')) {
      setFile(file);
      onFileSelect(file);
      const reader = new FileReader();
      reader.onload = (e) => {
        setPreview(e.target?.result as string);
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please upload an image file');
    }
  };

  const removeFile = () => {
    setFile(null);
    setPreview(null);
  };

  return (
    <div className="w-full max-w-xl mx-auto">
      {!file ? (
        <div
          className={`relative border-2 border-dashed rounded-xl p-8 transition-all duration-300 ease-in-out ${
            dragActive 
              ? 'border-indigo-500 bg-indigo-500/10' 
              : 'border-gray-700 hover:border-indigo-400 bg-gray-800/50'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            id="file-upload"
            className="hidden"
            onChange={handleChange}
            accept="image/*"
          />
          
          <label 
            htmlFor="file-upload" 
            className="flex flex-col items-center justify-center cursor-pointer"
          >
            <FileUp className={`w-16 h-16 mb-4 transition-colors duration-300 ${
              dragActive ? 'text-indigo-400' : 'text-gray-400'
            }`} />
            
            <p className="text-lg mb-2 font-medium text-gray-200">
              Drop your image here
            </p>
            <p className="text-sm text-gray-400 mb-4">
              or click to select a file
            </p>
            
            <button className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 transition-colors flex items-center">
              <Upload className="w-4 h-4 mr-2" />
              Choose File
            </button>
          </label>
        </div>
      ) : (
        <div className="bg-gray-800 rounded-xl overflow-hidden shadow-xl border border-gray-700">
          <div className="relative">
            <img 
              src={preview || ''} 
              alt="Preview" 
              className="w-full object-cover max-h-[300px]"
            />
            <button 
              onClick={removeFile}
              className="absolute top-2 right-2 bg-gray-900/80 p-2 rounded-full hover:bg-red-500 transition-colors"
            >
              <X className="w-5 h-5 text-white" />
            </button>
          </div>
          <div className="p-4">
            <p className="text-gray-300 text-sm truncate">{file.name}</p>
            <p className="text-gray-500 text-xs">
              {(file.size / 1024).toFixed(2)} KB
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileUploader;