type ButtonProps = {
    modelLoaded: boolean;
    onClick: () => void;
}

export const Button = ({ modelLoaded, onClick }: ButtonProps) => {
    return (
        <button
            onClick={onClick}
            className="rounded-full bg-[#2A1515] text-[#D4A6A6] py-4 px-6 focus:outline-none focus:ring-2 focus:ring-[#D4A6A6] focus:ring-opacity-50 hover:text-white"
        >
            {modelLoaded ? 'eject' : 'load model'}
        </button>
    );
};
