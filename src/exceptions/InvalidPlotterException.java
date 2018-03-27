package exceptions;

/**
 * Created by davidetalon on 02/08/17.
 */
public class InvalidPlotterException extends Exception{


    public InvalidPlotterException() {
        super();
    }

    public InvalidPlotterException(String message) {
        super(message);
    }

    public InvalidPlotterException(String message, Throwable cause) {
        super(message, cause);
    }

    public InvalidPlotterException(Throwable cause) {
        super(cause);
    }

}

